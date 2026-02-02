# trainers/train_loop.py
import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from evaluators.evaluator import Evaluator
from utils.serialization import save_checkpoint

def create_discriminator_optimizer(model, lr=1e-4):
    # 为对抗解耦模块中的判别器（属性判别器与域判别器）创建独立优化器
    disc_params = []
    if hasattr(model, 'adversarial_decoupler'):
        disc_params.extend(model.adversarial_decoupler.attr_disc.parameters())
        if hasattr(model.adversarial_decoupler, 'domain_disc'):
            disc_params.extend(model.adversarial_decoupler.domain_disc.parameters())
    
    if len(disc_params) > 0:
        return torch.optim.Adam(disc_params, lr=lr, betas=(0.5, 0.999))
    return None


def curriculum_train_epoch(trainer, train_loader, optimizer, optimizer_disc, epoch, total_batches):
    # 课程学习模式下的单个 Epoch 训练逻辑
    from utils.meters import AverageMeter
    
    trainer.model.train()
    loss_meters = {}
    
    # 获取并更新当前阶段的损失权重
    phase = trainer.curriculum.get_current_phase(epoch)
    weights = trainer.curriculum.get_loss_weights(epoch, trainer.performance_history)
    trainer.loss.update_weights(weights)
    
    # 阶段信息展示与对抗参数更新
    trainer.curriculum.print_phase_summary(epoch)
    if hasattr(trainer.model, 'adversarial_decoupler'):
        progress = (epoch - 1) / trainer.args.epochs
        trainer.model.adversarial_decoupler.update_lambda(progress)
    
    # 动态调整进度条宽度：优先尝试从 STDIN (fd=0) 获取宽度，以绕过 pipe/tee 的限制
    import shutil
    import os
    try:
        # 尝试从 stdin 获取真实的 TTY 宽度
        term_width = os.get_terminal_size(0).columns
    except OSError:
        # 如果失败（例如后台运行），回退到 shutil 检测
        term_width = shutil.get_terminal_size((80, 20)).columns
    
    tqdm_width = term_width

    for key in weights.keys():
        if key not in loss_meters:
            loss_meters[key] = AverageMeter()
    if 'total' not in loss_meters:
        loss_meters['total'] = AverageMeter()
    
    progress_bar = tqdm(
        train_loader, 
        desc=f"[Phase {phase}] [Epoch {epoch}/{trainer.args.epochs}]",
        ncols=tqdm_width, 
        leave=True,
        total=total_batches
    )
    
    for batch_idx, inputs in enumerate(progress_bar):
        # ---- 第一步：训练特征提取器（主模型） ----
        optimizer.zero_grad()
        
        # 计算特征提取损失
        loss_dict = trainer.run(inputs, epoch, batch_idx, total_batches, training_phase='feature')
        loss = loss_dict['total']
        
        # 异常检测与反向传播
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            if trainer.monitor:
                trainer.monitor.logger.error(f"❌ Epoch {epoch} Batch {batch_idx} 出现 NaN/Inf 损失，跳过该批次")
            continue
        
        if trainer.scaler:
            trainer.scaler.scale(loss).backward()
            has_grads = any(p.grad is not None for group in optimizer.param_groups for p in group['params'])
            if has_grads:
                trainer.scaler.unscale_(optimizer)
                trainer.clip_grad_norm_by_layer(trainer.model, max_norm=1.0)
                trainer.scaler.step(optimizer)
                trainer.scaler.update()
        else:
            loss.backward()
            trainer.clip_grad_norm_by_layer(trainer.model, max_norm=1.0)
            optimizer.step()
        
        # ---- 第二步：训练判别器（仅在解耦阶段开启，且按频率触发） ----
        if phase >= 2 and optimizer_disc is not None:
            if trainer.curriculum.should_train_discriminator(epoch, batch_idx, total_batches):
                optimizer_disc.zero_grad()
                loss_dict_disc = trainer.run(inputs, epoch, batch_idx, total_batches, training_phase='discriminator')
                loss_disc = loss_dict_disc['total']
                
                if not (torch.isnan(loss_disc).any() or torch.isinf(loss_disc).any()):
                    loss_disc.backward()
                    optimizer_disc.step()
                    for key in ['discriminator_attr', 'discriminator_domain']:
                        if key in loss_dict_disc and key in loss_meters:
                            loss_meters[key].update(loss_dict_disc[key].item())
        
        # 更新损失统计与进度条展示
        for key, val in loss_dict.items():
            if key in loss_meters:
                loss_meters[key].update(val.item() if isinstance(val, torch.Tensor) else val)
        
        display_loss = loss.item()
        progress_bar.set_postfix_str(f"loss: {display_loss:.4f}")
    
    progress_bar.close()
    return loss_meters


def train_with_curriculum(trainer, train_loader, query_loader, gallery_loader, checkpoint_dir):
    # 完整的课程学习训练流程控制器
    from trainers.trainer import EarlyStopping
    
    # 实例化优化器与调度器
    optimizer = torch.optim.AdamW(
        [p for p in trainer.model.parameters() if p.requires_grad],
        lr=trainer.args.lr,
        weight_decay=trainer.args.weight_decay
    )
    optimizer_disc = create_discriminator_optimizer(trainer.model, lr=trainer.args.lr * 0.5)
    
    from torch.optim.lr_scheduler import CosineAnnealingLR
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=trainer.args.epochs)
    early_stopping = EarlyStopping(patience=20, min_delta=0.001, logger=trainer.monitor)
    
    best_mAP = 0.0
    best_checkpoint_path = None
    total_batches = len(train_loader)
    
    # 遍历每个 Epoch
    for epoch in range(1, trainer.args.epochs + 1):
        loss_meters = curriculum_train_epoch(
            trainer, train_loader, optimizer, optimizer_disc, epoch, total_batches
        )
        
        # 学习率动态调整
        phase = trainer.curriculum.get_current_phase(epoch)
        lr_mult = trainer.curriculum.get_learning_rate_multiplier(epoch)
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_mult
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 定期执行模型评估
        if query_loader and gallery_loader:
            evaluator = Evaluator(trainer.model, args=trainer.args)
            metrics = evaluator.evaluate(
                query_loader, gallery_loader,
                query_loader.dataset.data, gallery_loader.dataset.data,
                checkpoint_path=None, epoch=epoch
            )
            
            current_mAP, current_rank1 = metrics['mAP'], metrics['rank1']
            print(f"\n{'='*60}\n📊 Epoch {epoch} [阶段 {phase}] 评估结果:\n"
                  f"  Rank-1: {metrics['rank1']:.3f} | Rank-5: {metrics['rank5']:.3f} | Rank-10: {metrics['rank10']:.3f} | mAP: {metrics['mAP']:.3f}\n{'='*60}\n")
            
            trainer.performance_history.append({
                'epoch': epoch, 'mAP': current_mAP, 'rank1': current_rank1,
                'rank5': metrics['rank5'], 'rank10': metrics['rank10']
            })
            
            # 早停检查与最佳模型保存
            early_stopping(current_mAP)
            if early_stopping.early_stop:
                if trainer.monitor: trainer.monitor.logger.info(f"性能连续未提升，在 Epoch {epoch} 触发早停")
                break
            
            if current_mAP > best_mAP:
                best_mAP = current_mAP
                if checkpoint_dir:
                    model_dir = Path(checkpoint_dir) / 'model'
                    model_dir.mkdir(parents=True, exist_ok=True)
                    dataset_name = trainer._get_dataset_name()
                    new_best_path = str(model_dir / f"best_{dataset_name}.pth")
                    
                    if best_checkpoint_path and Path(best_checkpoint_path).exists():
                        Path(best_checkpoint_path).unlink()
                    
                    save_checkpoint({
                        'model': trainer.model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch, 'mAP': current_mAP, 'phase': phase
                    }, fpath=new_best_path)
                    
                    best_checkpoint_path = new_best_path
                    if trainer.monitor:
                        trainer.monitor.logger.info(f"✅ 刷新最佳记录: mAP={best_mAP:.3f}, 模型已保存至 {best_checkpoint_path}")
        
        # 阶段自动过渡检测
        if trainer.curriculum.should_transition_phase(epoch, trainer.performance_history):
            if trainer.monitor: trainer.monitor.logger.info(f"🚀 性能达标，在 Epoch {epoch} 触发阶段自动过渡")
    
    # 训练结束总结
    import shutil
    width = min(max(shutil.get_terminal_size((80, 20)).columns, 80), 100)
    print(f"\n{'='*width}\n🎉 训练任务圆满完成！\n   最佳 mAP 指标: {best_mAP:.4f}\n"
          f"   模型检查点: {best_checkpoint_path}\n{'='*width}\n")
    
    return best_mAP, best_checkpoint_path
