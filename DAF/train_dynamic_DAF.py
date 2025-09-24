import argparse
import datetime
import torch.backends.cudnn as cudnn
import json
import yaml
from pathlib import Path
import torch
import os
import time
import numpy as np
from collections import OrderedDict

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from lib.datasets import build_dataset
from engine import train_one_epoch, evaluate
from lib.samplers import RASampler
from model.vision_transformer_timm import VisionTransformerSepQKV
from model.dynamic_DAF import DynamicSensitivityAnalyzer, DynamicDAFManager, create_dynamic_DAF_model

import model as models
from timm.models import load_checkpoint

try:
    # SLUMRM
    from mmcv.runner import init_dist
except ModuleNotFoundError as e:
    print(f'{e}. Cannot use multiple-node training...')

from timm.utils.clip_grad import dispatch_clip_grad
from lib import utils
from lib.utils import save_to_csv
from train_DAF import get_args_parser


def main(args):
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        init_dist(launcher=args.launcher)
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()

    print(args)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    device = torch.device(args.device)

    # 固定随机种子以确保可重复性
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args,)
    dataset_val, _ = build_dataset(is_train=False, args=args,)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    'Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')

            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=int(2 * args.batch_size),
        sampler=sampler_val, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False
    )

    print(f"{args.data_set} dataset, train: {len(dataset_train)}, evaluation: {len(dataset_val)}")

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    print('mixup_active', mixup_active)
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # 创建用于敏感度分析的数据加载器
    dataset_sensitivity, _ = build_dataset(is_train=True, args=args, )
    sampler_init = torch.utils.data.SequentialSampler(dataset_sensitivity)

    data_loader_sensitivity = torch.utils.data.DataLoader(
        dataset_sensitivity, sampler=sampler_init,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    # 确保初始敏感度文件存在
    if args.initial_sensitivity_path and not os.path.exists(args.initial_sensitivity_path):
        print(f"警告: 初始敏感度文件 {args.initial_sensitivity_path} 不存在!")
        
        # 尝试找到替代路径
        sensitivity_dir = os.path.dirname(args.initial_sensitivity_path)
        if not os.path.exists(sensitivity_dir):
            os.makedirs(sensitivity_dir, exist_ok=True)
            print(f"创建目录: {sensitivity_dir}")
        
        # 检查是否有dynamic_sensitivity目录中的文件可以复制
        dynamic_path = args.initial_sensitivity_path.replace('sensitivity_', 'dynamic_sensitivity_')
        dynamic_path = dynamic_path.replace('/param_req_', '/epoch_0/param_req_')
        
        if os.path.exists(dynamic_path):
            print(f"找到动态敏感度文件，复制到标准路径: {dynamic_path} -> {args.initial_sensitivity_path}")
            import shutil
            shutil.copy2(dynamic_path, args.initial_sensitivity_path)
        else:
            print(f"无法找到替代的敏感度文件，将进行初始敏感度分析")
            args.initial_sensitivity_path = None

    # 初始化模型
    if args.initial_sensitivity_path:
        # 使用初始敏感度配置创建模型
        model = create_dynamic_DAF_model(
            args=args,
            model_name=args.model_name,
            num_classes=args.nb_classes,
            sensitivity_path=args.initial_sensitivity_path
        )
    else:
        # 创建用于初始敏感度分析的基本模型
        model = create_dynamic_DAF_model(
            args=args,
            model_name=args.model_name,
            num_classes=args.nb_classes
        )

    # 加载预训练权重
    if args.resume:
        if '.pth' in args.resume:
            if args.resume.endswith('mae_pretrain_vit_base.pth'):
                state_dict = torch.load(args.resume, map_location='cpu')['model']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q')] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k')] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v')] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    else:
                        new_dict[name] = state_dict[name]

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from MAE model: ', msg)

            elif args.resume.endswith('linear-vit-b-300ep.pth'):
                state_dict = torch.load(args.resume, map_location='cpu')['state_dict']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q').split('module.')[1]] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k').split('module.')[1]] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v').split('module.')[1]] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    elif 'head.' in name:
                        continue
                    else:
                        new_dict[name.split('module.')[1]] = state_dict[name]

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from MoCo model: ', msg)

            elif args.resume.endswith('swin_base_patch4_window7_224_22k.pth'):
                state_dict = torch.load(args.resume, map_location='cpu')['model']
                new_dict = OrderedDict()
                for name in state_dict.keys():
                    if 'attn.qkv.' in name:
                        new_dict[name.replace('qkv', 'q')] = state_dict[name][:state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'k')] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                        new_dict[name.replace('qkv', 'v')] = state_dict[name][-state_dict[name].shape[0] // 3:]
                    elif 'head.' in name:
                        continue
                    else:
                        new_dict[name] = state_dict[name]

                if args.nb_classes != model.head.weight.shape[0]:
                    model.reset_classifier(args.nb_classes)

                msg = model.load_state_dict(new_dict, strict=False)
                print('Resuming from Swin model: ', msg)

            else:
                raise NotImplementedError

        else:
            load_checkpoint(model, args.resume)
            if args.nb_classes != model.head.weight.shape[0]:
                model.reset_classifier(args.nb_classes)

    model.to(device)
    model_ema = None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # 设置损失函数
    if args.mixup > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # 创建优化器
    optimizer = utils.build_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()
    lr_scheduler, _ = create_scheduler(args, optimizer)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    # 保存配置
    with open(output_dir / "config.yaml", 'w') as f:
        f.write(args_text)



    dynamic_DAF_manager = DynamicDAFManager(
        model=model,
        sensitivity_analyzer=sensitivity_analyzer,
        update_interval=args.dynamic_update_interval,
        param_budget=args.param_budget,
        dataset=args.data_set,
        exp_name=args.exp_name
    )

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, amp=args.amp)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        return

    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    # 如果没有初始敏感度配置，先进行一次敏感度分析
    if args.initial_sensitivity_path is None:
        print("执行初始敏感度分析...")
        param_dict = sensitivity_analyzer.analyze(
            dataset=args.data_set,
            epoch=0,
            structured_only=False
        )
        
        # 使用新生成的敏感度配置创建模型
        new_sensitivity_path = f'sensitivity_{args.exp_name}/{args.data_set}/param_req_{args.param_budget}.pth'
        if not os.path.exists(new_sensitivity_path):
            closest_budget = min(
                [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], 
                key=lambda x: abs(x - args.param_budget)
            )
            new_sensitivity_path = f'sensitivity_{args.exp_name}/{args.data_set}/param_req_{closest_budget}.pth'
            print(f"未找到目标预算配置，使用最接近的预算: {closest_budget}")
        
        # 创建新模型
        model = create_dynamic_DAF_model(
            args=args,
            model_name=args.model_name,
            num_classes=args.nb_classes,
            sensitivity_path=new_sensitivity_path
        )
        
        # 加载预训练权重
        if args.resume:
            if '.pth' in args.resume:
                if args.resume.endswith('mae_pretrain_vit_base.pth'):
                    state_dict = torch.load(args.resume, map_location='cpu')['model']
                    new_dict = OrderedDict()
                    for name in state_dict.keys():
                        if 'attn.qkv.' in name:
                            new_dict[name.replace('qkv', 'q')] = state_dict[name][:state_dict[name].shape[0] // 3]
                            new_dict[name.replace('qkv', 'k')] = state_dict[name][state_dict[name].shape[0] // 3:-state_dict[name].shape[0] // 3]
                            new_dict[name.replace('qkv', 'v')] = state_dict[name][-state_dict[name].shape[0] // 3:]
                        else:
                            new_dict[name] = state_dict[name]

                    msg = model.load_state_dict(new_dict, strict=False)
                    print('Resuming from MAE model: ', msg)
                else:
                    # 其他预训练模型的处理方式...
                    pass
        
        # 更新模型
        model.to(device)
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module
        else:
            model_without_ddp = model
        
        # 更新优化器
        optimizer = utils.build_optimizer(args, model_without_ddp)
        
        # 更新敏感度分析器的模型引用
        sensitivity_analyzer.model = model
        
        # 更新动态DAF管理器的模型引用
        dynamic_DAF_manager.model = model

    # 训练循环
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        # 更新动态DAF管理器的当前轮次
        dynamic_DAF_manager.current_epoch = epoch

        # 动态调整更新间隔
        if epoch < 251:
            dynamic_DAF_manager.update_interval = 10  # 初期(1-40轮)每5轮分析一次


        # 检查是否需要更新微调策略
        if epoch > 0 and dynamic_DAF_manager.should_update():  # 跳过第一轮，因为已经进行了初始化
            # 更新微调策略
            update_success = dynamic_DAF_manager.update_tuning_strategy()
            
            if update_success:
                # 如果策略更新成功，重新创建模型
                new_sensitivity_path = f'sensitivity_{args.exp_name}/{args.data_set}/param_req_{args.param_budget}.pth'
                if not os.path.exists(new_sensitivity_path):
                    closest_budget = min(
                        [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0], 
                        key=lambda x: abs(x - args.param_budget)
                    )
                    new_sensitivity_path = f'sensitivity_{args.exp_name}/{args.data_set}/param_req_{closest_budget}.pth'
                
                # 保存当前模型权重
                checkpoint_path = f'{args.output_dir}/checkpoint_epoch_{epoch}.pth'
                if utils.is_main_process():
                    torch.save({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
                
                # 获取旧模型的结构化配置
                old_structured_config = None
                if hasattr(model_without_ddp, 'structured_list'):
                    old_structured_config = {
                        'tuned_matrices': model_without_ddp.structured_list,
                        'tuned_vectors': getattr(model_without_ddp, 'tuned_vectors', [])
                    }
                
                # 创建新模型
                new_model = create_dynamic_DAF_model(
                    args=args,
                    model_name=args.model_name,
                    num_classes=args.nb_classes,
                    sensitivity_path=new_sensitivity_path,
                    old_structured_config=old_structured_config  # 传递旧模型的结构化配置
                )
                
                # 迁移知识（加载旧模型的权重）
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                msg = new_model.load_state_dict(checkpoint['model'], strict=False)
                print(f"知识迁移结果: {msg}")
                
                # 更新模型
                new_model.to(device)
                if args.distributed:
                    new_model = torch.nn.parallel.DistributedDataParallel(
                        new_model, device_ids=[args.gpu], find_unused_parameters=True
                    )
                    model_without_ddp = new_model.module
                else:
                    model_without_ddp = new_model
                
                model = new_model
                
                # 更新优化器
                optimizer = utils.build_optimizer(args, model_without_ddp)
                
                # 更新敏感度分析器的模型引用
                sensitivity_analyzer.model = model
                
                # 更新动态DAF管理器的模型引用
                dynamic_DAF_manager.model = model

        # 训练一个轮次
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            amp=args.amp, scaler=args.scaler
        )

        # 更新学习率
        lr_scheduler.step(epoch)

        # 评估
        if epoch % args.val_interval == 0 or epoch >= args.epochs-10:  # 在最后几轮更频繁地评估
            test_stats = evaluate(data_loader_val, model, device, amp=args.amp)
            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(
                f"[{args.exp_name}] Max accuracy on the {args.data_set} dataset {len(dataset_val)} with ({args.opt}, {args.lr}, {args.weight_decay}), {max_accuracy:.2f}%")

            # 保存到CSV
            save_to_csv('csvs/' + args.exp_name, args.data_set, "%.2f" % round(max_accuracy,2))

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,}

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # 保存检查点
        if args.output_dir and (epoch % 10 == 0 or epoch == args.epochs - 1):
            if utils.is_main_process():
                torch.save({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, f'{args.output_dir}/checkpoint_epoch_{epoch}.pth')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic DAF training and evaluation script', parents=[get_args_parser()])
    
    # 添加Dynamic DAF特有的参数
    parser.add_argument('--dynamic_update_interval', default=10, type=int, help='每隔多少轮更新一次敏感度分析和微调策略')
    parser.add_argument('--param_budget', default=0.4, type=float, help='参数预算（占总参数的比例）')
    parser.add_argument('--initial_sensitivity_path', default='', type=str, help='初始敏感度配置路径，如果为空则先进行敏感度分析')
    
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args) 