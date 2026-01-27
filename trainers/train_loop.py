# trainers/train_loop.py
"""
课程学习训练循环核心逻辑
用于替换trainer.py中的train方法
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from evaluators.evaluator import Evaluator
from utils.serialization import save_checkpoint


def create_discriminator_optimizer(model, lr=1e-4):
    """为判别器创建独立的优化器"""
    disc_params = []
    if hasattr(model, 'adversarial_decoupler'):
        disc_params.extend(model.adversarial_decoupler.attr_disc.parameters())
        if hasattr(model.adversarial_decoupler, 'domain_disc'):
            disc_params.extend(model.adversarial_decoupler.domain_disc.parameters())
    
    if len(disc_params) > 0:
        return torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.999))
    return None


def curriculum_train_epoch(trainer, train_loader, optimizer, optimizer_disc, epoch, total_batches):
    """
    课程学习单个epoch训练
    
    Args:
        trainer: Trainer实例
        train_loader: 训练数据加载器
        optimizer: 主优化器（特征提取器）
        optimizer_disc: 判别器优化器
        epoch: 当前epoch
        total_batches: 总batch数
    
    Returns:
        loss_meters: dict of AverageMeter
    """
    from utils.meters import AverageMeter
    
    trainer.model.train()
    loss_meters = {}
    
    # 获取当前阶段配置
    phase = trainer.curriculum.get_current_phase(epoch)
    weights = trainer.curriculum.get_loss_weights(epoch, trainer.performance_history)
    
    # 更新Loss权重
    trainer.loss.update_weights(weights)
    
    # 打印阶段摘要（每个epoch开始时）
    trainer.curriculum.print_phase_summary(epoch)
    
    # 更新对抗模块的lambda
    if hasattr(trainer.model, 'adversarial_decoupler'):
        progress = (epoch - 1) / trainer.args.epochs
        trainer.model.adversarial_decoupler.update_lambda(progress)
    
    # 初始化loss meters
    for key in weights.keys():
        if key not in loss_meters:
            loss_meters[key] = AverageMeter()
    if 'total' not in loss_meters:
        loss_meters['total'] = AverageMeter()
    
    # 训练循环
    progress_bar = tqdm(
        train_loader, 
        desc=f"[Phase {phase}] [Epoch {epoch}/{trainer.args.epochs}]",
        dynamic_ncols=True, 
        leave=True,
        total=total_batches
    )
    
    for batch_idx, inputs in enumerate(progress_bar):
        # ==== Step 1: 训练特征提取器 ====
        optimizer.zero_grad()
        
        # 前向传播 + 计算损失（training_phase='feature'）
        loss_dict = trainer.run(inputs, epoch, batch_idx, total_batches, training_phase='feature')
        loss = loss_dict['total']
        
        # NaN检测
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            if trainer.monitor:
                trainer.monitor.logger.error(f"❌ NaN/Inf loss at E{epoch} B{batch_idx}, skipping")
            continue
        
        # 反向传播
        if trainer.scaler:
            trainer.scaler.scale(loss).backward()
            
            # 检查梯度
            has_grads = any(p.grad is not None for group in optimizer.param_groups for p in group['params'])
            
            if has_grads:
                trainer.scaler.unscale_(optimizer)
                trainer.clip_grad_norm_by_layer(trainer.model, max_norm=1.0)  # 🔥 降低到1.0
                trainer.scaler.step(optimizer)
                trainer.scaler.update()
        else:
            loss.backward()
            trainer.clip_grad_norm_by_layer(trainer.model, max_norm=1.0)
            optimizer.step()
        
        # ==== Step 2: 训练判别器（Phase 2/3，每2个batch）====
        if phase >= 2 and optimizer_disc is not None:
            if trainer.curriculum.should_train_discriminator(epoch, batch_idx, total_batches):
                optimizer_disc.zero_grad()
                
                # 重新前向传播（training_phase='discriminator'）
                loss_dict_disc = trainer.run(inputs, epoch, batch_idx, total_batches, training_phase='discriminator')
                loss_disc = loss_dict_disc['total']
                
                if not (torch.isnan(loss_disc).any() or torch.isinf(loss_disc).any()):
                    loss_disc.backward()
                    optimizer_disc.step()
                    
                    # 记录判别器损失
                    for key in ['discriminator_attr', 'discriminator_domain']:
                        if key in loss_dict_disc and key in loss_meters:
                            loss_meters[key].update(loss_dict_disc[key].item())
        
        # ==== Step 3: 更新loss meters ====
        for key, val in loss_dict.items():
            if key in loss_meters:
                if isinstance(val, torch.Tensor):
                    loss_meters[key].update(val.item())
                else:
                    loss_meters[key].update(val)
        
        # ==== Step 4: 更新进度条 ====
        # 只显示主要损失
        progress_str = f"Loss: {loss.item():.4f}"
        if 'id_triplet' in loss_dict:
            progress_str += f" | Triplet: {loss_dict['id_triplet'].item():.4f}"
        progress_bar.set_postfix_str(progress_str)
    
    progress_bar.close()
    return loss_meters


def train_with_curriculum(trainer, train_loader, query_loader, gallery_loader, checkpoint_dir):
    """
    完整的课程学习训练流程
    
    Args:
        trainer: Trainer实例
        train_loader: 训练数据加载器
        query_loader: 查询集加载器
        gallery_loader: 图库加载器
        checkpoint_dir: 检查点保存目录
    """
    from trainers.trainer import EarlyStopping
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=trainer.args.lr,
        weight_decay=trainer.args.weight_decay
    )
    
    # 创建判别器优化器
    optimizer_disc = create_discriminator_optimizer(trainer.model, lr=trainer.args.lr * 0.5)
    
    # 学习率调度器
    from torch.optim.lr_scheduler import CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=trainer.args.epochs)
    
    # 早停
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, logger=trainer.monitor)
    
    # 训练状态
    best_mAP = 0.0
    best_checkpoint_path = None
    total_batches = len(train_loader)
    
    # 主训练循环
    for epoch in range(1, trainer.args.epochs + 1):
        # 训练一个epoch
        loss_meters = curriculum_train_epoch(
            trainer, train_loader, optimizer, optimizer_disc, epoch, total_batches
        )
        
        # 学习率调度（根据当前阶段动态调整）
        phase = trainer.curriculum.get_current_phase(epoch)
        lr_mult = trainer.curriculum.get_learning_rate_multiplier(epoch)
        
        # 更新学习率
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        
        # 清理显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # ==== 评估 ====
        if query_loader and gallery_loader:
            evaluator = Evaluator(trainer.model, args=trainer.args)
            metrics = evaluator.evaluate(
                query_loader, gallery_loader,
                query_loader.dataset.data,
                gallery_loader.dataset.data,
                checkpoint_path=None,
                epoch=epoch
            )
            
            current_mAP = metrics['mAP']
            current_rank1 = metrics['rank1']
            
            # 打印评估结果
            print(f"\n{'='*60}")
            print(f"📊 Epoch {epoch} [Phase {phase}] Evaluation:")
            print(f"  mAP:     {metrics['mAP']:.4f}")
            print(f"  Rank-1:  {metrics['rank1']:.4f}")
            print(f"  Rank-5:  {metrics['rank5']:.4f}")
            print(f"  Rank-10: {metrics['rank10']:.4f}")
            print(f"{'='*60}\n")
            
            # 记录性能历史
            trainer.performance_history.append({
                'epoch': epoch,
                'mAP': current_mAP,
                'rank1': current_rank1,
                'rank5': metrics['rank5'],
                'rank10': metrics['rank10']
            })
            
            # 早停检查
            early_stopping(current_mAP)
            if early_stopping.early_stop:
                if trainer.monitor:
                    trainer.monitor.logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # 保存最佳模型
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                
                if checkpoint_dir:
                    ckpt_dir_path = Path(checkpoint_dir)
                    model_dir = ckpt_dir_path / 'model'
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    dataset_name = trainer._get_dataset_name()
                    new_best_checkpoint_path = str(model_dir / f"best_{dataset_name}.pth")
                    
                    # 删除旧checkpoint
                    if best_checkpoint_path and Path(best_checkpoint_path).exists():
                        Path(best_checkpoint_path).unlink()
                    
                    # 保存新checkpoint
                    save_checkpoint({
                        'model': trainer.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'mAP': current_mAP,
                        'phase': phase
                    }, fpath=new_best_checkpoint_path)
                    
                    best_checkpoint_path = new_best_checkpoint_path
                    
                    if trainer.monitor:
                        trainer.monitor.logger.info(f"✅ New best: mAP={best_mAP:.4f}, saved to {best_checkpoint_path}")
        
        # 检查是否需要提前过渡阶段
        if trainer.curriculum.should_transition_phase(epoch, trainer.performance_history):
            if trainer.monitor:
                trainer.monitor.logger.info(f"🚀 Phase transition triggered at epoch {epoch}")
    
    # 训练完成
    print(f"\n{'='*70}")
    print(f"🎉 Training Completed!")
    print(f"   Best mAP: {best_mAP:.4f}")
    if best_checkpoint_path:
        print(f"   Best Model: {best_checkpoint_path}")
    print(f"{'='*70}\n")
    
    return best_mAP, best_checkpoint_path
