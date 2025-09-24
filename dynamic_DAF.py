"""
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
"""



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
vit_operation_dict = {'q': 0, 'k': 1, 'v': 2, 'proj': 3, 'fc1': 4, 'fc2': 5}

class DynamicSensitivityAnalyzer:
    def __init__(self, model, criterion, data_loader, device, 
                 alpha=10., beta=5., batch_num=8, low_rank_dim=8,     
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
        self.history_sensitivity = None
        
    def analyze(self, dataset, epoch, structured_only=False, param_budget=0.2):
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
            elif 'adapter' in name:
                adapter_count += 1 
            elif '.attn.q.bias' in name or '.attn.k.bias' in name or '.attn.v.bias' in name:
                attn_bias_count += 1                
            elif 'norm' in name and len(param.shape) == 1:
                norm_count += 1
            elif len(param.shape) == 1:
                other_vector_count += 1
            elif len(param.shape) >= 2:
                matrix_count += 1       
        param_dict = {
            'special_active_params': [],
            'special_frozen_params': []
        }     
        self.model.train()
        self.criterion.train()      
        torch.manual_seed(0)
        np.random.seed(0)      
        grad_dict = {}
        for name, param in self.model.named_parameters():       
            grad_dict[name] = torch.zeros_like(param, dtype=torch.float32, device=self.device)
        amp_enabled = torch.is_autocast_enabled()
        if amp_enabled:
            torch.cuda.amp.autocast(enabled=False)
        for idx, (samples, targets) in enumerate(self.data_loader):
            if idx >= self.batch_num:
                break    
            samples = samples.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            peft_params = []
            original_requires_grad_state = {}
            for name, param in self.model.named_parameters():
                if 'structured' in name or 'adapter' in name or 'lora' in name:
                    peft_params.append((name, param))
                    original_requires_grad_state[name] = param.requires_grad
                    param.requires_grad = False 
            self.model.zero_grad()
            with torch.cuda.amp.autocast(enabled=False):
                outputs = self.model(samples)
                loss = self.criterion(outputs, targets)
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                for name, param in peft_params:
                    param.requires_grad = original_requires_grad_state[name]
                if amp_enabled:
                    torch.cuda.amp.autocast(enabled=True)
                return None
            loss.backward()
            for name, param in peft_params:
                param.requires_grad = original_requires_grad_state[name]
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_float = param.grad.to(torch.float32)
                    param_float = param.data.to(device=grad_float.device, dtype=grad_float.dtype)
                    grad_dict[name] += torch.abs(grad_float * param_float).detach()
            torch.cuda.synchronize()
        if amp_enabled:
            torch.cuda.amp.autocast(enabled=True)
        if self.history_sensitivity is not None:
            smoothing_factor = 0.7 
            for name in grad_dict:
                if name in self.history_sensitivity:
                    if isinstance(self.history_sensitivity[name], torch.Tensor):
                        grad_dict[name] = smoothing_factor * self.history_sensitivity[name] + \
                                         (1 - smoothing_factor) * grad_dict[name]
        self.history_sensitivity = copy.deepcopy(grad_dict)
        grad_skip_kwd_list = ['head', 'cls_token', 'patch_embed', 'pos_embed']
        grad_matrix_kwd_list = ['.q.', '.k.', '.v.', 'proj', 'fc']
        grad_vector_kwd_list = ['norm', 'bias']
        grad_shapes = {}
        for key in grad_dict.keys():
            if not any(kwd in key for kwd in grad_skip_kwd_list):   
                if not isinstance(grad_dict[key], torch.Tensor): 
                    continue 
                grad_shapes[key] = grad_dict[key].shape
        if not grad_shapes
            return None
        structured_params = []  
        vector_params = []     
        special_params = []   
        for key in grad_shapes.keys():
            sensitivity = grad_dict[key].sum().item()
            param_size = np.prod(grad_shapes[key])
            if 'sparse_weight' in key or 'adapter' in key or '.attn.q.bias' in key or '.attn.k.bias' in key or '.attn.v.bias' in key:
                special_params.append((key, sensitivity, param_size))
            elif any(kwd in key for kwd in grad_vector_kwd_list) and len(grad_shapes[key]) == 1: 
                if 'norm' in key and sensitivity > 0.3:
                    special_params.append((key, sensitivity, param_size))
                else:
                    vector_params.append((key, sensitivity, param_size))   
            elif any(kwd in key for kwd in grad_matrix_kwd_list) and len(grad_shapes[key]) >= 2:
                in_dim = grad_shapes[key][1]
                out_dim = grad_shapes[key][0]
                structured_size = self.get_structured_param_num(
                    structured_type=self.structured_type,
                    low_rank_dim=self.low_rank_dim,
                    in_dim=in_dim,
                    out_dim=out_dim
                )
                structured_params.append((key, sensitivity, structured_size))
        total_structured_size = sum(size for _, _, size in structured_params)
        total_vector_size = sum(size for _, _, size in vector_params)
        total_special_size = sum(size for _, _, size in special_params)
        total_param_size = total_structured_size + total_vector_size + total_special_size
        param_budgets = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05]
        budget_results = {}
        structured_ratio = 0.3  
        vector_ratio = 0.4      
        special_ratio = 0.3     
        for budget in param_budgets:
            structured_budget = budget * structured_ratio * total_param_size
            vector_budget = budget * vector_ratio * total_param_size
            special_budget = budget * special_ratio * total_param_size
            sorted_structured = sorted(structured_params, key=lambda x: x[1], reverse=True)
            sorted_vector = sorted(vector_params, key=lambda x: x[1], reverse=True)
            sorted_special = sorted(special_params, key=lambda x: x[1], reverse=True)
            selected_structured = []
            current_count = 0
            for key, sensitivity, size in sorted_structured:
                if current_count + size <= structured_budget:
                    selected_structured.append((key, sensitivity))
                    current_count += size  
                else:
                    break
            selected_vector = []
            current_count = 0
            for key, sensitivity, size in sorted_vector:
                if current_count + size <= vector_budget:
                    selected_vector.append((key, sensitivity))
                    current_count += size
                 
                else:
                    break
            selected_special = []
            current_count = 0
            for key, sensitivity, size in sorted_special:
                if current_count + size <= special_budget:
                    selected_special.append((key, sensitivity))
                    current_count += size 
                else:
                    break
            structured_names = [key for key, _ in selected_structured]
            tuned_vectors = [key for key, _ in selected_vector]
            special_active_params = [key for key, _ in selected_special]
            special_frozen_params = [key for key, _, _ in sorted_special if key not in special_active_params]
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
                    continue
            structured_param_num = sum(size for key, _, size in structured_params if key in structured_names)
            vector_param_num = sum(size for key, _, size in vector_params if key in tuned_vectors)
            special_param_num = sum(size for key, _, size in special_params if key in special_active_params)
            active_matrix_count = sum(sum(row) for row in tuned_matrices)
            total_matrix_count = len(tuned_matrices) * len(tuned_matrices[0])
            total_params = (structured_param_num + vector_param_num + special_param_num) / 1e6
            vector_percent = vector_param_num/total_vector_size*100 if total_vector_size > 0 else 0
            special_percent = special_param_num/total_special_size*100 if total_special_size > 0 else 0
            res = {
                'params': total_params,
                'structured_params': structured_param_num / 1e6,
                'vector_params': vector_param_num / 1e6,
                'special_params': special_param_num / 1e6,
                'tuned_matrices': tuned_matrices,
                'tuned_vectors': tuned_vectors,
                'special_active_params': special_active_params,
                'special_frozen_params': special_frozen_params,
                'unstructured_name_shapes': {},
                'unstructured_name_shapes_int': {},
                'unstructured_params': 0,
                'unstructured_indexes': torch.zeros(0).long(),
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
            if epoch == 0:
                save_dir = f'sensitivity_{self.exp_name}/{dataset}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_on_master(res, f'{save_dir}/param_req_{budget}.pth')
            else:
                save_dir = f'dynamic_sensitivity_{self.exp_name}/{dataset}/epoch_{epoch}'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_on_master(res, f'{save_dir}/param_req_{budget}.pth')
                save_dir_compat = f'sensitivity_{self.exp_name}/{dataset}'
                if not os.path.exists(save_dir_compat):
                    os.makedirs(save_dir_compat)
                save_on_master(res, f'{save_dir_compat}/param_req_{budget}.pth')
            budget_results[budget] = res
        param_dict['special_active_params'] = budget_results[param_budget]['special_active_params']
        param_dict['special_frozen_params'] = budget_results[param_budget]['special_frozen_params']
        actual_params = {k: v['params'] for k, v in budget_results.items()}
        return param_dict
    
    def get_structured_param_num(self, structured_type=None, in_dim=768, out_dim=768, low_rank_dim=8):
        if structured_type == 'lora':
            return in_dim * low_rank_dim + low_rank_dim * out_dim
        elif structured_type == 'adapter':
            return out_dim * low_rank_dim + low_rank_dim * out_dim + low_rank_dim + out_dim
        else:
            raise NotImplementedError

class DynamicDAFManager:
    def __init__(self, model, sensitivity_analyzer, update_interval=10, 
                 param_budget=0.2, dataset='cifar', exp_name='dynamic_DAF'):
        self.model = model
        self.sensitivity_analyzer = sensitivity_analyzer
        self.update_interval = update_interval
        self.param_budget = param_budget
        self.dataset = dataset
        self.exp_name = exp_name
        self.current_epoch = 0
        self.current_config = None
    
    def should_update(self):
        if self.current_epoch == 0:
            return True
        should_update = self.current_epoch % self.update_interval == 0
        if should_update:
            print({self.current_epoch}, {self.update_interval})
        return should_update
    
    def update_tuning_strategy(self):
        param_dict = self.sensitivity_analyzer.analyze(
            dataset=self.dataset,
            epoch=self.current_epoch,
            structured_only=False,
            param_budget=self.param_budget
        )
        
        if param_dict is None 
            return False
        
        new_config_path = f'sensitivity_{self.sensitivity_analyzer.exp_name}/{self.dataset}/param_req_{self.param_budget}.pth'
        if not os.path.exists(new_config_path):
            param_budgets = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
            closest_budget = min(param_budgets, key=lambda x: abs(x - self.param_budget))
            new_config_path = f'sensitivity_{self.sensitivity_analyzer.exp_name}/{self.dataset}/param_req_{closest_budget}.pth'
         
        new_config = torch.load(new_config_path, map_location='cpu')
        new_config['special_active_params'] = param_dict['special_active_params']
        new_config['special_frozen_params'] = param_dict['special_frozen_params']
        
        if self.current_config is None:
            self.current_config = new_config
            return True
        if self.should_apply_new_config(self.current_config, new_config):
            old_config = self.current_config
            self.current_config = new_config
            self.transfer_knowledge(old_config, new_config)
            return True
        else: 
            return False
    
    def should_apply_new_config(self, old_config, new_config):
        old_matrices = old_config['tuned_matrices']
        new_matrices = new_config['tuned_matrices']
        matrix_changes = 0
        total_matrices = 0
        for i in range(len(old_matrices)):
            for j in range(len(old_matrices[i])):
                if old_matrices[i][j] != new_matrices[i][j]:
                    matrix_changes += 1
                total_matrices += 1
        
   
        old_vectors = set(old_config['tuned_vectors'])
        new_vectors = set(new_config['tuned_vectors'])
        vector_changes = len(old_vectors.symmetric_difference(new_vectors))
        total_vectors = len(old_vectors.union(new_vectors))
        old_unstructured = old_config.get('unstructured_params', 0)
        new_unstructured = new_config.get('unstructured_params', 0)
        unstructured_change_ratio = abs(old_unstructured - new_unstructured) / max(old_unstructured, new_unstructured, 1e-10)
        old_special_active = set(old_config.get('special_active_params', []))
        new_special_active = set(new_config.get('special_active_params', []))
        old_special_frozen = set(old_config.get('special_frozen_params', []))
        new_special_frozen = set(new_config.get('special_frozen_params', []))
        special_active_changes = len(old_special_active.symmetric_difference(new_special_active))
        special_frozen_changes = len(old_special_frozen.symmetric_difference(new_special_frozen))
        total_special = len(old_special_active) + len(old_special_frozen) + len(new_special_active) + len(new_special_frozen)
        special_change_ratio = 0 if total_special == 0 else (special_active_changes + special_frozen_changes) / total_special
        matrix_change_ratio = matrix_changes / total_matrices if total_matrices > 0 else 0
        vector_change_ratio = vector_changes / total_vectors if total_vectors > 0 else 0
        matrix_threshold = 0.3
        vector_threshold = 0.4  
        special_threshold = 0.3  
        old_stats = old_config.get('stats', {})
        if old_stats:
            print({old_stats.get('active_matrix_count', 0)}/{old_stats.get('total_matrix_count', 0)})
            print( {old_stats.get('active_matrix_params', 0)/1e6:.6f}M/{old_stats.get('total_matrix_params', 0)/1e6:.6f}M ({old_stats.get('active_matrix_params', 0)/max(1, old_stats.get('total_matrix_params', 0))*100:.2f}%))
            print({len(old_config['tuned_vectors'])})
            print({old_stats.get('active_vector_params', 0)/1e6:.6f}/{old_stats.get('total_vector_params', 0)/1e6:.6f} ({old_stats.get('active_vector_params', 0)/max(1, old_stats.get('total_vector_params', 0))*100:.2f}%))
            print({old_config.get('unstructured_params', 0)/1e6:.6f})
            print({len(old_special_active)})
            print({len(old_special_frozen)})
            print({old_config['params']:.6f})
        else:
            print( {sum(sum(row) for row in old_matrices)}/{len(old_matrices)*len(old_matrices[0])} )
            print({len(old_config['tuned_vectors'])} )
            print({old_config.get('unstructured_params', 0)/1e6:.6f})
            print({len(old_special_active)})
            print( {len(old_special_frozen)})
            print( {old_config['params']:.6f})
        
        new_stats = new_config.get('stats', {})
        if new_stats:
            print({new_stats.get('active_matrix_count', 0)}/{new_stats.get('total_matrix_count', 0)} )
            print( {new_stats.get('active_matrix_params', 0)/1e6:.6f}/{new_stats.get('total_matrix_params', 0)/1e6:.6f}M ({new_stats.get('active_matrix_params', 0)/max(1, new_stats.get('total_matrix_params', 0))*100:.2f}%))
            print({len(new_config['tuned_vectors'])} )
            print( {new_stats.get('active_vector_params', 0)/1e6:.6f}/{new_stats.get('total_vector_params', 0)/1e6:.6f}M ({new_stats.get('active_vector_params', 0)/max(1, new_stats.get('total_vector_params', 0))*100:.2f}%))
            print( {new_config.get('unstructured_params', 0)/1e6:.6f})
            print({len(new_special_active)})
            print({len(new_special_frozen)})
            print({new_config['params']:.6f})
        else:
            print({sum(sum(row) for row in new_matrices)}/{len(new_matrices)*len(new_matrices[0])})
            print( {len(new_config['tuned_vectors'])})
            print({new_config.get('unstructured_params', 0)/1e6:.6f})
            print({len(new_special_active)})
            print({len(new_special_frozen)})
            print({new_config['params']:.6f})
        return (matrix_change_ratio > matrix_threshold or 
                vector_change_ratio > vector_threshold or 
                unstructured_change_ratio > unstructured_threshold or
                special_change_ratio > special_threshold)
    
    def transfer_knowledge(self, old_config, new_config):
        
        self.analyze_config_changes(old_config, new_config)
    
    def analyze_config_changes(self, old_config, new_config):
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
        
       
        old_vectors = set(old_config['tuned_vectors'])
        new_vectors = set(new_config['tuned_vectors'])
        added_vectors = list(new_vectors - old_vectors)
        removed_vectors = list(old_vectors - new_vectors)
        old_special_active = set(old_config.get('special_active_params', []))
        new_special_active = set(new_config.get('special_active_params', []))
        old_special_frozen = set(old_config.get('special_frozen_params', []))
        new_special_frozen = set(new_config.get('special_frozen_params', []))
        added_special_active = list(new_special_active - old_special_active)
        removed_special_active = list(old_special_active - new_special_active)
        added_special_frozen = list(new_special_frozen - old_special_frozen)
        removed_special_frozen = list(old_special_frozen - new_special_frozen)
        old_params = old_config['params']
        new_params = new_config['params']
        print( {new_params - old_params:.6f}M ({(new_params - old_params) / old_params * 100:.2f}%))
        old_structured = old_config.get('structured_params', 0)
        new_structured = new_config.get('structured_params', 0)
        old_vector = old_config.get('vector_params', 0)
        new_vector = new_config.get('vector_params', 0)
        old_special = old_config.get('special_params', 0)
        new_special = new_config.get('special_params', 0)
      

def create_dynamic_DAF_model(args, model_name, num_classes, sensitivity_path=None, old_structured_config=None):
    if sensitivity_path:
        param_info = torch.load(sensitivity_path, map_location='cpu')
        tuned_vectors = param_info['tuned_vectors']
        tuned_matrices = param_info['tuned_matrices']
        frozen_matrices = None
        if old_structured_config is not None:
            old_tuned_matrices = old_structured_config['tuned_matrices']
            merged_matrices = []
            frozen_matrices = [] 
            for i in range(len(tuned_matrices)):
                merged_row = []
                frozen_row = []
                for j in range(len(tuned_matrices[i])):
                    if tuned_matrices[i][j] == 1:
                        merged_row.append(1)
                        frozen_row.append(False)
                    elif old_tuned_matrices[i][j] == 1:
                        merged_row.append(1)
                        frozen_row.append(True)
                    else:
                        merged_row.append(0)
                        frozen_row.append(False)
                merged_matrices.append(merged_row)
                frozen_matrices.append(frozen_row)
            tuned_matrices = merged_matrices
        else:
            print('Use both structured and unstructured tuning simultaneously')
        
        fully_fine_tuned_keys = []
        fully_fine_tuned_keys.extend(tuned_vectors)
        fully_fine_tuned_keys.extend(['head.weight', 'head.bias', 'cls_token'])
        unstructured_name_shapes = param_info.get('unstructured_name_shapes', {})
        unstructured_indexes = param_info.get('unstructured_indexes', torch.zeros(0).long())
        unstructured_params = param_info.get('unstructured_params', 0)
        if unstructured_name_shapes is None:
            unstructured_name_shapes = {}
        special_active_params = param_info.get('special_active_params', [])
        special_frozen_params = param_info.get('special_frozen_params', [])
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
                grad_mask = None
        
        model = VisionTransformerDAF(
            img_size=args.input_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            freeze_backbone=args.freeze_stage,
            structured_list=tuned_matrices,
            frozen_matrices=frozen_matrices,  
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
          model = VisionTransformerDAF(
            img_size=args.input_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            freeze_backbone=args.freeze_stage,
            num_classes=num_classes
        )
    return model 
