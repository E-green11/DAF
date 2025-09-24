import torch
import torch.nn as nn
import numpy as np
import copy
import math
from collections import OrderedDict
import os
import time
from tqdm import tqdm
from lib.utils import save_on_master


# This script implements the core logic of Dynamic Adaptive Fine-tuning (DAF),
# a novel paradigm for parameter-efficient fine-tuning.
#
# Key innovations of the DAF framework implemented in this file include:
# 1. DynamicDAFManager: Orchestrates the PERIODIC sensitivity analysis during training,
#    enabling the model's trainable structure to adapt dynamically.
# 2. DynamicSensitivityAnalyzer: Performs a CONTEXT-AWARE analysis on the evolving
#    model to identify the most critical parameters at each stage of learning.
# 3. create_dynamic_DAF_model: Implements the "REBUILD-AND-REFOCUS" strategy. This
#    function dynamically reconstructs the model based on the latest sensitivity
#    analysis, preserving learned knowledge in outdated modules by freezing them.



# 全局变量，与engine.py中保持一致
vit_operation_dict = {'q': 0, 'k': 1, 'v': 2, 'proj': 3, 'fc1': 4, 'fc2': 5}


class DynamicSensitivityAnalyzer:
    """
    动态敏感度分析器：周期性评估模型参数的敏感度
    """
    def __init__(self, model, criterion, data_loader, device, 
                 alpha=10., beta=5., batch_num=8, low_rank_dim=8,
                 structured_type='lora', structured_vector=True, exp_name='dynamic_DAF'):
        """
        初始化动态敏感度分析器
        
        Args:
            model: 当前模型
            criterion: 损失函数
            data_loader: 数据加载器
            device: 计算设备
            alpha: 结构化调优的敏感度阈值
            beta: 向量结构化调优的敏感度阈值
            batch_num: 用于敏感度分析的批次数
            low_rank_dim: 低秩维度
            structured_type: 结构化调优类型 ('lora' 或 'adapter')
            structured_vector: 是否对向量参数进行结构化调优
            exp_name: 实验名称
        """
        self.model = model
        self.criterion = criterion
        self.data_loader = data_loader
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.batch_num = batch_num
        self.low_rank_dim = low_rank_dim
        self.structured_type = structured_type
        self.structured_vector = structured_vector
        self.exp_name = exp_name
        
        # 保存历史敏感度信息，用于平滑过渡
        self.history_sensitivity = None
        
    def analyze(self, dataset, epoch, structured_only=False, param_budget=0.2):
        """
        分析当前模型的参数敏感度
        
        Args:
            dataset: 数据集名称
            epoch: 当前训练轮次
            structured_only: 是否只使用结构化调优
            param_budget: 参数预算比例
            
        Returns:
            dict: 包含敏感度分析结果的字典
        """
        print(f"===== 执行动态敏感度分析，轮次 {epoch} =====")
        
        # 打印模型参数统计
        print("===== 模型参数统计 =====")
        total_params = 0
        sparse_weight_count = 0
        adapter_count = 0
        attn_bias_count = 0
        norm_count = 0
        other_vector_count = 0
        matrix_count = 0
        
        for name, param in self.model.named_parameters():
            param_size = np.prod(param.shape)
            total_params += param_size
            
            if 'sparse_weight' in name:
                sparse_weight_count += 1
                print(f"sparse_weight参数: {name}, 形状: {param.shape}, 大小: {param_size}")
            elif 'adapter' in name:
                adapter_count += 1
                print(f"adapter参数: {name}, 形状: {param.shape}, 大小: {param_size}")
            elif '.attn.q.bias' in name or '.attn.k.bias' in name or '.attn.v.bias' in name:
                attn_bias_count += 1
                print(f"注意力偏置参数: {name}, 形状: {param.shape}, 大小: {param_size}")
            elif 'norm' in name and len(param.shape) == 1:
                norm_count += 1
            elif len(param.shape) == 1:
                other_vector_count += 1
            elif len(param.shape) >= 2:
                matrix_count += 1
        
        print(f"总参数数量: {total_params/1e6:.6f}M")
        print(f"sparse_weight参数数量: {sparse_weight_count}")
        print(f"adapter参数数量: {adapter_count}")
        print(f"注意力偏置参数数量: {attn_bias_count}")
        print(f"norm参数数量: {norm_count}")
        print(f"其他向量参数数量: {other_vector_count}")
        print(f"矩阵参数数量: {matrix_count}")


        # 初始化结果字典
        param_dict = {
            'special_active_params': [],
            'special_frozen_params': []
        }
        
        # 设置模型为训练模式
        self.model.train()
        self.criterion.train()
        
        # 设置固定随机种子以确保可重复性
        torch.manual_seed(0)
        np.random.seed(0)
        
        # 初始化梯度字典，确保所有值都是张量
        grad_dict = {}
        for name, param in self.model.named_parameters():
            # 为每个参数创建一个与参数形状相同的零张量
            grad_dict[name] = torch.zeros_like(param, dtype=torch.float32, device=self.device)
        
        # 临时禁用自动混合精度，确保使用相同的数据类型
        amp_enabled = torch.is_autocast_enabled()
        if amp_enabled:
            torch.cuda.amp.autocast(enabled=False)
        
        # 在数据子集上累积梯度
        for idx, (samples, targets) in enumerate(self.data_loader):
            print(f'===== 敏感度分析批次: {idx}')
            if idx >= self.batch_num:
                break
                
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            #识别并临时冻结PEFT模块，以实现解耦分析
            peft_params = []
            original_requires_grad_state = {}
            for name, param in self.model.named_parameters():
                # 根据您PEFT模块的命名规则定义关键字
                if 'structured' in name or 'adapter' in name or 'lora' in name:
                    peft_params.append((name, param))
                    original_requires_grad_state[name] = param.requires_grad
                    param.requires_grad = False  # 临时冻结**

            self.model.zero_grad()
            
            # 前向传播和反向传播 - 禁用混合精度
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model(samples)
                loss = self.criterion(outputs, targets)
                
            loss_value = loss.item()
            
            if not math.isfinite(loss_value):
                print(f"损失值为 {loss_value}，停止训练")
                for name, param in peft_params:
                    param.requires_grad = original_requires_grad_state[name]
                # 恢复自动混合精度状态
                if amp_enabled:
                    torch.cuda.amp.autocast(enabled=True)
                return None
                
            # 确保所有参数使用相同的数据类型进行反向传播
            loss.backward()
            for name, param in peft_params:
                param.requires_grad = original_requires_grad_state[name]
            # 累积梯度
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # 确保梯度是float32类型
                    grad_float = param.grad.to(torch.float32)
                    param_float = param.data.to(device=grad_float.device, dtype=grad_float.dtype)
                    grad_dict[name] += torch.abs(grad_float * param_float).detach()
            torch.cuda.synchronize()
        
        # 恢复自动混合精度状态
        if amp_enabled:
            torch.cuda.amp.autocast(enabled=True)
        
        # 如果有历史敏感度，进行平滑更新
        if self.history_sensitivity is not None:
            smoothing_factor = 0.7  # 平滑因子
            for name in grad_dict:
                if name in self.history_sensitivity:
                    # 确保历史敏感度是张量
                    if isinstance(self.history_sensitivity[name], torch.Tensor):
                        grad_dict[name] = smoothing_factor * self.history_sensitivity[name] + \
                                         (1 - smoothing_factor) * grad_dict[name]
        
        # 保存当前敏感度作为历史记录
        self.history_sensitivity = copy.deepcopy(grad_dict)
        

        grad_skip_kwd_list = ['head', 'cls_token', 'patch_embed', 'pos_embed']
        grad_matrix_kwd_list = ['.q.', '.k.', '.v.', 'proj', 'fc']
        grad_vector_kwd_list = ['norm', 'bias']
        
        # 收集需要分析的参数形状
        grad_shapes = {}
        
        for key in grad_dict.keys():
            if not any(kwd in key for kwd in grad_skip_kwd_list):
                # 确保grad_dict[key]是张量
                if not isinstance(grad_dict[key], torch.Tensor):
                    print(f"警告: {key}的梯度不是张量，而是{type(grad_dict[key])}，跳过")
                    continue
                
                grad_shapes[key] = grad_dict[key].shape
        
        # 检查是否有有效的梯度形状
        if not grad_shapes:
            print("错误: 没有找到有效的梯度形状，无法继续敏感度分析")
            return None
            
        # ====== 新的参数选择逻辑 ======
        # 收集并分类所有参数及其敏感度
        structured_params = []  # 结构化参数（矩阵）
        vector_params = []      # 结构化参数（向量）
        special_params = []     # 特殊一维参数
        
        print("分类参数...")
        for key in grad_shapes.keys():
            # 计算参数敏感度
            sensitivity = grad_dict[key].sum().item()
            param_size = np.prod(grad_shapes[key])
            
            # 扩大特殊参数的识别范围
            if 'sparse_weight' in key or 'adapter' in key or '.attn.q.bias' in key or '.attn.k.bias' in key or '.attn.v.bias' in key:
                special_params.append((key, sensitivity, param_size))
                print(f"特殊参数: {key}, 敏感度: {sensitivity:.2f}, 大小: {param_size}")
            elif any(kwd in key for kwd in grad_vector_kwd_list) and len(grad_shapes[key]) == 1:
                # 将一些重要的向量参数归类为特殊参数
                if 'norm' in key and sensitivity > 0.3:
                    special_params.append((key, sensitivity, param_size))
                    print(f"特殊参数(高敏感度向量): {key}, 敏感度: {sensitivity:.2f}, 大小: {param_size}")
                else:
                    vector_params.append((key, sensitivity, param_size))
                    print(f"向量参数: {key}, 敏感度: {sensitivity:.2f}, 大小: {param_size}")
            elif any(kwd in key for kwd in grad_matrix_kwd_list) and len(grad_shapes[key]) >= 2:
                # 计算结构化参数数量
                in_dim = grad_shapes[key][1]
                out_dim = grad_shapes[key][0]
                structured_size = self.get_structured_param_num(
                    structured_type=self.structured_type,
                    low_rank_dim=self.low_rank_dim,
                    in_dim=in_dim,
                    out_dim=out_dim
                )
                structured_params.append((key, sensitivity, structured_size))
                print(f"矩阵参数: {key}, 敏感度: {sensitivity:.2f}, 结构化大小: {structured_size}")
        
        # 计算各类参数总量
        total_structured_size = sum(size for _, _, size in structured_params)
        total_vector_size = sum(size for _, _, size in vector_params)
        total_special_size = sum(size for _, _, size in special_params)
        total_param_size = total_structured_size + total_vector_size + total_special_size
        
        print(f"总参数统计:")
        print(f"  - 矩阵参数: {total_structured_size/1e6:.6f}M")
        print(f"  - 向量参数: {total_vector_size/1e6:.6f}M")
        print(f"  - 特殊一维参数: {total_special_size/1e6:.6f}M")
        print(f"  - 总参数: {total_param_size/1e6:.6f}M")
        
        # 为不同预算点强制选择参数
        param_budgets = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        budget_results = {}
        
        # 预算分配比例
        structured_ratio = 0.3  # 30%给矩阵
        vector_ratio = 0.4      # 40%给向量
        special_ratio = 0.3     # 30%给特殊参数
        
        print("按预算选择参数...")
        for budget in param_budgets:
            # 计算各类型参数的预算
            structured_budget = budget * structured_ratio * total_param_size
            vector_budget = budget * vector_ratio * total_param_size
            special_budget = budget * special_ratio * total_param_size
            
            print(f"\n预算 {budget}:")
            print(f"  - 结构化矩阵预算: {structured_budget/1e6:.6f}M ({structured_ratio*100:.1f}%)")
            print(f"  - 结构化向量预算: {vector_budget/1e6:.6f}M ({vector_ratio*100:.1f}%)")
            print(f"  - 特殊参数预算: {special_budget/1e6:.6f}M ({special_ratio*100:.1f}%)")
            
            # 按敏感度排序
            sorted_structured = sorted(structured_params, key=lambda x: x[1], reverse=True)
            sorted_vector = sorted(vector_params, key=lambda x: x[1], reverse=True)
            sorted_special = sorted(special_params, key=lambda x: x[1], reverse=True)
            
            # 选择结构化矩阵参数
            selected_structured = []
            current_count = 0
            for key, sensitivity, size in sorted_structured:
                if current_count + size <= structured_budget:
                    selected_structured.append((key, sensitivity))
                    current_count += size
                    print(f"选择结构化矩阵: {key}, 敏感度: {sensitivity:.2f}")
                else:
                    break
            
            # 选择结构化向量参数
            selected_vector = []
            current_count = 0
            for key, sensitivity, size in sorted_vector:
                if current_count + size <= vector_budget:
                    selected_vector.append((key, sensitivity))
                    current_count += size
                    print(f"选择结构化向量: {key}, 敏感度: {sensitivity:.2f}")
                else:
                    break
            
            # 选择特殊参数
            selected_special = []
            current_count = 0
            for key, sensitivity, size in sorted_special:
                if current_count + size <= special_budget:
                    selected_special.append((key, sensitivity))
                    current_count += size
                    print(f"选择特殊参数: {key}, 敏感度: {sensitivity:.2f}")
                else:
                    break
            
            # 处理选中的参数
            structured_names = [key for key, _ in selected_structured]
            tuned_vectors = [key for key, _ in selected_vector]
            special_active_params = [key for key, _ in selected_special]
            special_frozen_params = [key for key, _, _ in sorted_special if key not in special_active_params]
            
            # 构建tuned_matrices
            tuned_matrices = [[0, 0, 0, 0, 0, 0] for _ in range(12)]
            for name in structured_names:
                attr = name.split('.')
                if len(attr) != 5:
                    continue
                try:
                    block_idx = int(attr[1])
                    operation_idx = int(vit_operation_dict[attr[3]])
                    tuned_matrices[block_idx][operation_idx] = 1
                except (IndexError, ValueError, KeyError) as e:
                    print(f"警告: 处理结构化矩阵 {name} 时出错: {e}")
                    continue
            
            # 计算实际使用的参数量
            structured_param_num = sum(size for key, _, size in structured_params if key in structured_names)
            vector_param_num = sum(size for key, _, size in vector_params if key in tuned_vectors)
            special_param_num = sum(size for key, _, size in special_params if key in special_active_params)
            
            # 计算激活矩阵数量
            active_matrix_count = sum(sum(row) for row in tuned_matrices)
            total_matrix_count = len(tuned_matrices) * len(tuned_matrices[0])
            
            # 打印详细的参数统计信息
            print(f"\n===== 预算点 {budget} 的详细参数统计 =====")
            total_params = (structured_param_num + vector_param_num + special_param_num) / 1e6
            print(f"总参数预算: {budget} ({budget*100:.1f}%), 实际使用: {total_params:.6f}M ({total_params/(total_param_size/1e6)*100:.2f}%)")
            print(f"结构化矩阵: 激活 {active_matrix_count}/{total_matrix_count} 个矩阵, 参数量 {structured_param_num/1e6:.6f}M/{total_structured_size/1e6:.6f}M ({structured_param_num/total_structured_size*100:.2f}%)")
            vector_percent = vector_param_num/total_vector_size*100 if total_vector_size > 0 else 0
            print(f"结构化向量: 激活 {len(tuned_vectors)} 个向量, 参数量 {vector_param_num/1e6:.6f}M/{total_vector_size/1e6:.6f}M ({vector_percent:.2f}%)")
            special_percent = special_param_num/total_special_size*100 if total_special_size > 0 else 0
            print(f"特殊一维参数: 激活 {special_param_num/1e6:.6f}M/{total_special_size/1e6:.6f}M ({special_percent:.2f}%)")
            print(f"特殊活跃参数数量: {len(special_active_params)}")
            print(f"特殊冻结参数数量: {len(special_frozen_params)}")
            
            # 准备保存的结果
            res = {
                'params': total_params,
                'structured_params': structured_param_num / 1e6,
                'vector_params': vector_param_num / 1e6,
                'special_params': special_param_num / 1e6,
                'tuned_matrices': tuned_matrices,
                'tuned_vectors': tuned_vectors,
                'special_active_params': special_active_params,
                'special_frozen_params': special_frozen_params,
                # 为兼容性保留的字段
                'unstructured_name_shapes': {},
                'unstructured_name_shapes_int': {},
                'unstructured_params': 0,
                'unstructured_indexes': torch.zeros(0).long(),
                # 添加详细的统计信息
                'stats': {
                    'total_matrix_params': total_structured_size,
                    'active_matrix_params': structured_param_num,
                    'total_vector_params': total_vector_size,
                    'active_vector_params': vector_param_num,
                    'total_special_params': total_special_size,
                    'active_special_params': special_param_num,
                    'active_matrix_count': active_matrix_count,
                    'total_matrix_count': total_matrix_count
                }
            }
            
            # 创建保存目录
            if epoch == 0:
                # 初始敏感度分析保存到sensitivity_目录
                save_dir = f'sensitivity_{self.exp_name}/{dataset}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print(f'创建文件夹: {save_dir}')
                
                # 保存结果
                save_on_master(res, f'{save_dir}/param_req_{budget}.pth')
            else:
                # 动态更新的敏感度保存到dynamic_sensitivity_目录
                save_dir = f'dynamic_sensitivity_{self.exp_name}/{dataset}/epoch_{epoch}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print(f'创建文件夹: {save_dir}')
                
                # 保存结果
                save_on_master(res, f'{save_dir}/param_req_{budget}.pth')
                
                # 同时复制到sensitivity_目录，确保train_dynamic_DAF.py可以找到
                save_dir_compat = f'sensitivity_{self.exp_name}/{dataset}'
                if not os.path.exists(save_dir_compat):
                    os.makedirs(save_dir_compat)
                
                save_on_master(res, f'{save_dir_compat}/param_req_{budget}.pth')
            
            # 保存结果到字典
            budget_results[budget] = res
        
        # 更新param_dict
        param_dict['special_active_params'] = budget_results[param_budget]['special_active_params']
        param_dict['special_frozen_params'] = budget_results[param_budget]['special_frozen_params']
        
        # 打印预算信息
        actual_params = {k: v['params'] for k, v in budget_results.items()}
        print(f'预算: 实际参数: {actual_params}')
        
        return param_dict
    
    def get_structured_param_num(self, structured_type=None, in_dim=768, out_dim=768, low_rank_dim=8):
        """计算结构化调优参数数量"""
        if structured_type == 'lora':
            return in_dim * low_rank_dim + low_rank_dim * out_dim
        elif structured_type == 'adapter':
            return out_dim * low_rank_dim + low_rank_dim * out_dim + low_rank_dim + out_dim
        else:
            raise NotImplementedError


class DynamicDAFManager:
    """
    动态DAF管理器：管理动态敏感度分析和微调策略更新
    """
    def __init__(self, model, sensitivity_analyzer, update_interval=10, 
                 param_budget=0.2, dataset='cifar', exp_name='dynamic_DAF'):
        """
        初始化动态DAF管理器
        
        Args:
            model: 当前模型
            sensitivity_analyzer: 敏感度分析器
            update_interval: 更新间隔（轮次）
            param_budget: 参数预算（占总参数的比例）
            dataset: 数据集名称
            exp_name: 实验名称
        """
        self.model = model
        self.sensitivity_analyzer = sensitivity_analyzer
        self.update_interval = update_interval
        self.param_budget = param_budget
        self.dataset = dataset
        self.exp_name = exp_name
        self.current_epoch = 0
        self.current_config = None
    
    def should_update(self):
        """检查是否应该更新微调策略"""
        # 第一轮总是需要更新
        if self.current_epoch == 0:
            return True
        # 之后按照更新间隔检查
        should_update = self.current_epoch % self.update_interval == 0
        if should_update:
            print(f"当前轮次: {self.current_epoch}, 当前更新间隔: {self.update_interval}, 执行敏感度分析")
        return should_update
    
    def update_tuning_strategy(self):
        """更新微调策略"""
        print(f"===== 更新微调策略，轮次 {self.current_epoch} =====")
        
        # 执行敏感度分析
        param_dict = self.sensitivity_analyzer.analyze(
            dataset=self.dataset,
            epoch=self.current_epoch,
            structured_only=False,
            param_budget=self.param_budget
        )
        
        if param_dict is None:
            print("敏感度分析失败，保持当前配置")
            return False
        
        # 获取新配置路径 - 使用sensitivity_目录而不是dynamic_sensitivity_目录
        new_config_path = f'sensitivity_{self.sensitivity_analyzer.exp_name}/{self.dataset}/param_req_{self.param_budget}.pth'
        
        if not os.path.exists(new_config_path):
            # 尝试找到最接近的参数预算
            param_budgets = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            closest_budget = min(param_budgets, key=lambda x: abs(x - self.param_budget))
            new_config_path = f'sensitivity_{self.sensitivity_analyzer.exp_name}/{self.dataset}/param_req_{closest_budget}.pth'
            print(f"未找到目标预算配置，使用最接近的预算: {closest_budget}")
        
        # 加载新配置
        new_config = torch.load(new_config_path, map_location='cpu')
        
        # 添加特殊参数信息到新配置
        new_config['special_active_params'] = param_dict['special_active_params']
        new_config['special_frozen_params'] = param_dict['special_frozen_params']
        
        # 如果是首次配置，直接应用
        if self.current_config is None:
            self.current_config = new_config
            return True
        
        # 比较新旧配置，决定是否需要更新
        if self.should_apply_new_config(self.current_config, new_config):
            # 应用新配置前，保存当前配置用于知识迁移
            old_config = self.current_config
            self.current_config = new_config
            
            # 执行知识迁移
            self.transfer_knowledge(old_config, new_config)
            return True
        else:
            print("新配置与当前配置相似，保持当前配置")
            return False
    
    def should_apply_new_config(self, old_config, new_config):
        """
        判断是否应该应用新配置
        
        Args:
            old_config: 旧配置
            new_config: 新配置
            
        Returns:
            bool: 是否应该应用新配置
        """
        # 计算结构化调优配置的变化
        old_matrices = old_config['tuned_matrices']
        new_matrices = new_config['tuned_matrices']
        
        matrix_changes = 0
        total_matrices = 0
        
        for i in range(len(old_matrices)):
            for j in range(len(old_matrices[i])):
                if old_matrices[i][j] != new_matrices[i][j]:
                    matrix_changes += 1
                total_matrices += 1
        
        # 计算向量调优配置的变化
        old_vectors = set(old_config['tuned_vectors'])
        new_vectors = set(new_config['tuned_vectors'])
        
        vector_changes = len(old_vectors.symmetric_difference(new_vectors))
        total_vectors = len(old_vectors.union(new_vectors))
        
        # 计算非结构化调优配置的变化
        old_unstructured = old_config.get('unstructured_params', 0)
        new_unstructured = new_config.get('unstructured_params', 0)
        
        unstructured_change_ratio = abs(old_unstructured - new_unstructured) / max(old_unstructured, new_unstructured, 1e-10)
        
        # 计算特殊参数配置的变化
        old_special_active = set(old_config.get('special_active_params', []))
        new_special_active = set(new_config.get('special_active_params', []))
        old_special_frozen = set(old_config.get('special_frozen_params', []))
        new_special_frozen = set(new_config.get('special_frozen_params', []))
        
        special_active_changes = len(old_special_active.symmetric_difference(new_special_active))
        special_frozen_changes = len(old_special_frozen.symmetric_difference(new_special_frozen))
        total_special = len(old_special_active) + len(old_special_frozen) + len(new_special_active) + len(new_special_frozen)
        
        special_change_ratio = 0 if total_special == 0 else (special_active_changes + special_frozen_changes) / total_special
        
        # 综合评估变化程度
        matrix_change_ratio = matrix_changes / total_matrices if total_matrices > 0 else 0
        vector_change_ratio = vector_changes / total_vectors if total_vectors > 0 else 0
        
        # 设置变化阈值
        matrix_threshold = 0.1  # 10%的矩阵配置变化
        vector_threshold = 0.2  # 20%的向量配置变化
        unstructured_threshold = 0.3  # 30%的非结构化参数变化
        special_threshold = 0.2  # 20%的特殊参数变化
        
        # 打印详细的参数统计信息
        print(f"\n===== 配置比较详细信息 =====")
        
        # 打印旧配置统计信息
        print(f"旧配置参数统计:")
        old_stats = old_config.get('stats', {})
        if old_stats:
            print(f"  - 结构化矩阵: 激活 {old_stats.get('active_matrix_count', 0)}/{old_stats.get('total_matrix_count', 0)} 个矩阵")
            print(f"  - 结构化矩阵参数: {old_stats.get('active_matrix_params', 0)/1e6:.6f}M/{old_stats.get('total_matrix_params', 0)/1e6:.6f}M ({old_stats.get('active_matrix_params', 0)/max(1, old_stats.get('total_matrix_params', 0))*100:.2f}%)")
            print(f"  - 结构化向量: 激活 {len(old_config['tuned_vectors'])} 个向量")
            print(f"  - 结构化向量参数: {old_stats.get('active_vector_params', 0)/1e6:.6f}M/{old_stats.get('total_vector_params', 0)/1e6:.6f}M ({old_stats.get('active_vector_params', 0)/max(1, old_stats.get('total_vector_params', 0))*100:.2f}%)")
            print(f"  - 非结构化参数: {old_config.get('unstructured_params', 0)/1e6:.6f}M")
            print(f"  - 特殊活跃参数: {len(old_special_active)}")
            print(f"  - 特殊冻结参数: {len(old_special_frozen)}")
            print(f"  - 总参数: {old_config['params']:.6f}M")
        else:
            print(f"  - 结构化矩阵: 激活 {sum(sum(row) for row in old_matrices)}/{len(old_matrices)*len(old_matrices[0])} 个矩阵")
            print(f"  - 结构化向量: 激活 {len(old_config['tuned_vectors'])} 个向量")
            print(f"  - 非结构化参数: {old_config.get('unstructured_params', 0)/1e6:.6f}M")
            print(f"  - 特殊活跃参数: {len(old_special_active)}")
            print(f"  - 特殊冻结参数: {len(old_special_frozen)}")
            print(f"  - 总参数: {old_config['params']:.6f}M")
        
        # 打印新配置统计信息
        print(f"新配置参数统计:")
        new_stats = new_config.get('stats', {})
        if new_stats:
            print(f"  - 结构化矩阵: 激活 {new_stats.get('active_matrix_count', 0)}/{new_stats.get('total_matrix_count', 0)} 个矩阵")
            print(f"  - 结构化矩阵参数: {new_stats.get('active_matrix_params', 0)/1e6:.6f}M/{new_stats.get('total_matrix_params', 0)/1e6:.6f}M ({new_stats.get('active_matrix_params', 0)/max(1, new_stats.get('total_matrix_params', 0))*100:.2f}%)")
            print(f"  - 结构化向量: 激活 {len(new_config['tuned_vectors'])} 个向量")
            print(f"  - 结构化向量参数: {new_stats.get('active_vector_params', 0)/1e6:.6f}M/{new_stats.get('total_vector_params', 0)/1e6:.6f}M ({new_stats.get('active_vector_params', 0)/max(1, new_stats.get('total_vector_params', 0))*100:.2f}%)")
            print(f"  - 非结构化参数: {new_config.get('unstructured_params', 0)/1e6:.6f}M")
            print(f"  - 特殊活跃参数: {len(new_special_active)}")
            print(f"  - 特殊冻结参数: {len(new_special_frozen)}")
            print(f"  - 总参数: {new_config['params']:.6f}M")
        else:
            print(f"  - 结构化矩阵: 激活 {sum(sum(row) for row in new_matrices)}/{len(new_matrices)*len(new_matrices[0])} 个矩阵")
            print(f"  - 结构化向量: 激活 {len(new_config['tuned_vectors'])} 个向量")
            print(f"  - 非结构化参数: {new_config.get('unstructured_params', 0)/1e6:.6f}M")
            print(f"  - 特殊活跃参数: {len(new_special_active)}")
            print(f"  - 特殊冻结参数: {len(new_special_frozen)}")
            print(f"  - 总参数: {new_config['params']:.6f}M")
        
        # 打印变化比例
        print(f"矩阵变化比例: {matrix_change_ratio:.2f}, 阈值: {matrix_threshold}")
        print(f"向量变化比例: {vector_change_ratio:.2f}, 阈值: {vector_threshold}")
        print(f"非结构化参数变化比例: {unstructured_change_ratio:.2f}, 阈值: {unstructured_threshold}")
        print(f"特殊参数变化比例: {special_change_ratio:.2f}, 阈值: {special_threshold}")
        
        # 如果任何一个变化超过阈值，则应用新配置
        return (matrix_change_ratio > matrix_threshold or 
                vector_change_ratio > vector_threshold or 
                unstructured_change_ratio > unstructured_threshold or
                special_change_ratio > special_threshold)
    
    def transfer_knowledge(self, old_config, new_config):
        """
        在配置变更时迁移知识
        
        注意：这个函数是一个占位符，实际的知识迁移需要在模型加载新配置时实现
        真正的知识迁移会在train_dynamic_DAF.py中实现
        
        Args:
            old_config: 旧配置
            new_config: 新配置
        """
        print("准备知识迁移...")
        # 分析配置变化
        self.analyze_config_changes(old_config, new_config)
    
    def analyze_config_changes(self, old_config, new_config):
        """
        分析配置变化
        
        Args:
            old_config: 旧配置
            new_config: 新配置
        """
        # 分析结构化矩阵变化
        old_matrices = old_config['tuned_matrices']
        new_matrices = new_config['tuned_matrices']
        
        added_matrices = []
        removed_matrices = []
        
        for i in range(len(old_matrices)):
            for j in range(len(old_matrices[i])):
                if old_matrices[i][j] == 0 and new_matrices[i][j] == 1:
                    added_matrices.append((i, j))
                elif old_matrices[i][j] == 1 and new_matrices[i][j] == 0:
                    removed_matrices.append((i, j))
        
        # 分析结构化向量变化
        old_vectors = set(old_config['tuned_vectors'])
        new_vectors = set(new_config['tuned_vectors'])
        
        added_vectors = list(new_vectors - old_vectors)
        removed_vectors = list(old_vectors - new_vectors)
        
        # 分析特殊参数变化
        old_special_active = set(old_config.get('special_active_params', []))
        new_special_active = set(new_config.get('special_active_params', []))
        old_special_frozen = set(old_config.get('special_frozen_params', []))
        new_special_frozen = set(new_config.get('special_frozen_params', []))
        
        added_special_active = list(new_special_active - old_special_active)
        removed_special_active = list(old_special_active - new_special_active)
        added_special_frozen = list(new_special_frozen - old_special_frozen)
        removed_special_frozen = list(old_special_frozen - new_special_frozen)
        
        # 打印变化摘要
        print(f"配置变化摘要:")
        print(f"  - 添加的结构化矩阵: {len(added_matrices)}")
        print(f"  - 移除的结构化矩阵: {len(removed_matrices)}")
        print(f"  - 添加的结构化向量: {len(added_vectors)}")
        print(f"  - 移除的结构化向量: {len(removed_vectors)}")
        print(f"  - 非结构化参数变化: {new_config.get('unstructured_params', 0) - old_config.get('unstructured_params', 0):.6f}M")
        print(f"  - 新激活的特殊参数: {len(added_special_active)}")
        print(f"  - 不再激活的特殊参数: {len(removed_special_active)}")
        print(f"  - 新冻结的特殊参数: {len(added_special_frozen)}")
        print(f"  - 不再冻结的特殊参数: {len(removed_special_frozen)}")
        
        # 打印总参数变化
        old_params = old_config['params']
        new_params = new_config['params']
        print(f"  - 总参数变化: {new_params - old_params:.6f}M ({(new_params - old_params) / old_params * 100:.2f}%)")
        
        # 打印各类参数的变化
        old_structured = old_config.get('structured_params', 0)
        new_structured = new_config.get('structured_params', 0)
        old_vector = old_config.get('vector_params', 0)
        new_vector = new_config.get('vector_params', 0)
        old_special = old_config.get('special_params', 0)
        new_special = new_config.get('special_params', 0)
        
        print(f"  - 结构化矩阵参数变化: {new_structured - old_structured:.6f}M")
        print(f"  - 结构化向量参数变化: {new_vector - old_vector:.6f}M")
        print(f"  - 特殊参数变化: {new_special - old_special:.6f}M")


def create_dynamic_DAF_model(args, model_name, num_classes, sensitivity_path=None, old_structured_config=None):
    """
    创建动态DAF模型
    
    Args:
        args: 命令行参数
        model_name: 模型名称
        num_classes: 类别数量
        sensitivity_path: 敏感度配置路径
        old_structured_config: 旧模型的结构化参数配置，用于保留低敏感度参数
        
    Returns:
        model: 创建的模型
    """
    # 如果提供了敏感度路径，加载配置
    if sensitivity_path:
        param_info = torch.load(sensitivity_path, map_location='cpu')
        tuned_vectors = param_info['tuned_vectors']
        tuned_matrices = param_info['tuned_matrices']
        
        # 初始化冻结矩阵标记
        frozen_matrices = None
        
        # 如果提供了旧配置，合并配置
        if old_structured_config is not None:
            old_tuned_matrices = old_structured_config['tuned_matrices']
            # 创建合并后的矩阵配置
            merged_matrices = []
            frozen_matrices = []  # 记录哪些矩阵是冻结的
            
            for i in range(len(tuned_matrices)):
                merged_row = []
                frozen_row = []
                for j in range(len(tuned_matrices[i])):
                    if tuned_matrices[i][j] == 1:
                        # 新配置中需要调优
                        merged_row.append(1)
                        frozen_row.append(False)
                    elif old_tuned_matrices[i][j] == 1:
                        # 旧配置中有，新配置中不需要，保留但冻结
                        merged_row.append(1)
                        frozen_row.append(True)
                    else:
                        # 两者都不需要
                        merged_row.append(0)
                        frozen_row.append(False)
                merged_matrices.append(merged_row)
                frozen_matrices.append(frozen_row)
            
            # 使用合并后的配置
            tuned_matrices = merged_matrices
            
            print('使用合并的结构化调优配置，保留旧模型中的低敏感度参数')
        else:
            print('同时使用结构化和非结构化调优')
        
        fully_fine_tuned_keys = []
        fully_fine_tuned_keys.extend(tuned_vectors)
        fully_fine_tuned_keys.extend(['head.weight', 'head.bias', 'cls_token'])
        
        # 设置非结构化调优
        unstructured_name_shapes = param_info.get('unstructured_name_shapes', {})
        unstructured_indexes = param_info.get('unstructured_indexes', torch.zeros(0).long())
        unstructured_params = param_info.get('unstructured_params', 0)
        
        # 确保unstructured_name_shapes不为None
        if unstructured_name_shapes is None:
            unstructured_name_shapes = {}
            
        # 获取特殊参数
        special_active_params = param_info.get('special_active_params', [])
        special_frozen_params = param_info.get('special_frozen_params', [])
        
        # 处理非结构化调优掩码
        grad_mask = None
        if unstructured_params != 0 and unstructured_name_shapes and len(unstructured_name_shapes) > 0:
            try:
                grad_mask = torch.cat(
                    [torch.zeros(unstructured_name_shapes[key]).flatten() for key in unstructured_name_shapes.keys()])
                grad_mask[unstructured_indexes] = 1.
                grad_mask = grad_mask.split([np.cumprod(list(shape))[-1] for shape in unstructured_name_shapes.values()])
                grad_mask = {k: (mask.view(v) != 0).nonzero() for mask, (k, v) in
                             zip(grad_mask, unstructured_name_shapes.items())}
            except Exception as e:
                print(f"警告: 处理非结构化调优掩码时出错: {e}")
                grad_mask = None
        
        # 创建模型
        model = VisionTransformerDAF(
            img_size=args.input_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            freeze_backbone=args.freeze_stage,
            structured_list=tuned_matrices,
            frozen_matrices=frozen_matrices,  # 添加冻结矩阵标记
            tuned_vectors=tuned_vectors,
            low_rank_dim=args.low_rank_dim,
            block=args.block,
            num_classes=num_classes,
            structured_type=args.structured_type,
            structured_bias=args.structured_vector,
            unstructured_indexes=grad_mask,
            unstructured_shapes=unstructured_name_shapes,
            fully_fine_tuned_keys=fully_fine_tuned_keys,
            no_structured_drop_out=args.no_structured_drop_out,
            no_structured_drop_path=args.no_structured_drop_path,
            special_active_params=special_active_params,
            special_frozen_params=special_frozen_params,
        )
    else:
        # 创建用于敏感度分析的基本模型
        model = VisionTransformerDAF(
            img_size=args.input_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            freeze_backbone=args.freeze_stage,
            num_classes=num_classes
        )
    
    return model 
