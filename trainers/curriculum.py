# trainers/curriculum.py
"""
Curriculum Learning Scheduler
课程学习调度器：三阶段训练策略
"""

import torch
import torch.nn as nn


class CurriculumScheduler:
    """
    三阶段课程学习调度器
    
    Phase 1 (Epoch 1-15): ID-First Training
        - 目标：建立基础ID判别能力
        - 策略：禁用解耦损失，全力优化ID分类
        - 预期：Rank-1 > 30%
    
    Phase 2 (Epoch 16-40): Adversarial Decoupling
        - 目标：引入对抗式解耦
        - 策略：逐步增大对抗损失权重，使用GRL
        - 预期：Rank-1 > 50%, 解耦质量提升
    
    Phase 3 (Epoch 41+): Fine-tuning
        - 目标：精细化调整所有模块
        - 策略：平衡所有损失，优化融合模块
        - 预期：Rank-1 > 60%
    """
    
    def __init__(self, total_epochs=80, logger=None):
        self.total_epochs = total_epochs
        self.logger = logger
        
        # 阶段边界
        self.phase1_end = 15
        self.phase2_end = 40
        
        # 基础损失权重（Phase 1）
        # 🔥 修复：重新平衡损失权重
        self.base_weights = {
            'info_nce': 1.0,
            'id_triplet': 2.0,  # 🔥 从10.0降到2.0，避免主导训练
            'cloth_semantic': 0.1,  # 🔥 从0.01升到0.1，增强服装语义学习
            'spatial_orthogonal': 0.0,  # Phase 1禁用
            'semantic_alignment': 0.0,  # Phase 1禁用
            'ortho_reg': 0.0,
            'adversarial_attr': 0.0,  # Phase 1完全禁用
            'adversarial_domain': 0.0,
            'discriminator_attr': 0.0,
            'discriminator_domain': 0.0
        }
        
        if logger:
            logger.debug_logger.info("=" * 70)
            logger.debug_logger.info("📚 Curriculum Learning Scheduler Initialized")
            logger.debug_logger.info("=" * 70)
            logger.debug_logger.info(f"Phase 1 (Epoch 1-{self.phase1_end}): ID-First Training")
            logger.debug_logger.info(f"Phase 2 (Epoch {self.phase1_end+1}-{self.phase2_end}): Adversarial Decoupling")
            logger.debug_logger.info(f"Phase 3 (Epoch {self.phase2_end+1}+): Fine-tuning")
            logger.debug_logger.info("=" * 70)
    
    def get_current_phase(self, epoch):
        """获取当前训练阶段"""
        if epoch <= self.phase1_end:
            return 1
        elif epoch <= self.phase2_end:
            return 2
        else:
            return 3
    
    def get_loss_weights(self, epoch, performance_history=None):
        """
        动态获取损失权重
        
        Args:
            epoch: 当前epoch
            performance_history: dict, 历史性能指标 {'epoch': X, 'mAP': Y, 'rank1': Z}
        
        Returns:
            weights: dict, 损失权重
        """
        phase = self.get_current_phase(epoch)
        
        if phase == 1:
            # Phase 1: ID-First Training
            # 🔥 修复：重新平衡损失权重
            weights = {
                'info_nce': 1.0,
                'id_triplet': 2.0,  # 🔥 从10.0降到2.0
                'cloth_semantic': 0.1,  # 🔥 从0.01升到0.1
                'spatial_orthogonal': 0.0,
                'semantic_alignment': 0.0,
                'ortho_reg': 0.0,
                'adversarial_attr': 0.0,  # Phase 1完全禁用对抗
                'adversarial_domain': 0.0,
                'discriminator_attr': 0.0,
                'discriminator_domain': 0.0
            }

            # 动态调整：如果Rank-1已经超过30%，提前进入Phase 2
            if performance_history and len(performance_history) > 0:
                latest_rank1 = performance_history[-1].get('rank1', 0.0)
                if latest_rank1 > 0.30 and epoch >= 10:
                    if self.logger:
                        self.logger.logger.info(f"🎯 Early Phase Transition: Rank-1={latest_rank1:.1%} > 30%, advancing to Phase 2")
                    return self.get_loss_weights(self.phase1_end + 1, performance_history)

        elif phase == 2:
            # Phase 2: Adversarial Decoupling
            # 线性增加对抗损失权重
            progress = (epoch - self.phase1_end) / (self.phase2_end - self.phase1_end)

            # 🔥 修复：平滑过渡，降低对抗损失权重
            weights = {
                'info_nce': 1.0,
                'id_triplet': 2.0 - progress * 0.5,  # 从2.0降到1.5
                'cloth_semantic': 0.1 + progress * 0.2,  # 从0.1升到0.3
                'spatial_orthogonal': progress * 0.3,  # 从0升到0.3
                'semantic_alignment': progress * 0.05,  # 从0升到0.05
                'ortho_reg': progress * 0.2,  # 从0升到0.2
                'adversarial_attr': progress * 0.3,  # 🔥 从0升到0.3（从1.0大幅降低）
                'adversarial_domain': progress * 0.1,  # 从0升到0.1
                'discriminator_attr': 0.5,  # 🔥 降低判别器权重
                'discriminator_domain': 0.2  # 🔥 降低判别器权重
            }

            # 检查性能停滞
            if performance_history and len(performance_history) >= 5:
                recent_maps = [h.get('mAP', 0.0) for h in performance_history[-5:]]
                if max(recent_maps) - min(recent_maps) < 0.01:  # mAP变化<1%
                    if self.logger:
                        self.logger.logger.warning(f"⚠️  Performance plateau detected in Phase 2, adjusting weights")
                    # 增强ID学习
                    weights['id_triplet'] *= 1.2
                    weights['adversarial_attr'] *= 0.5

        else:
            # Phase 3: Fine-tuning
            # 🔥 修复：平衡所有损失，降低对抗损失权重
            weights = {
                'info_nce': 1.0,
                'id_triplet': 1.5,  # 保持适中
                'cloth_semantic': 0.3,  # 提升服装语义
                'spatial_orthogonal': 0.3,  # 降低
                'semantic_alignment': 0.05,  # 降低
                'ortho_reg': 0.2,
                'adversarial_attr': 0.2,  # 🔥 大幅降低
                'adversarial_domain': 0.05,  # 降低
                'discriminator_attr': 0.3,  # 降低
                'discriminator_domain': 0.1  # 降低
            }
        
        return weights
    
    def get_learning_rate_multiplier(self, epoch):
        """
        获取学习率倍数
        
        Phase 1: 1.0x (快速学习ID)
        Phase 2: 0.5x (稳定解耦)
        Phase 3: 0.3x (精细调整)
        """
        phase = self.get_current_phase(epoch)
        
        if phase == 1:
            return 1.0
        elif phase == 2:
            return 0.5
        else:
            return 0.3
    
    def should_train_discriminator(self, epoch, batch_idx, total_batches):
        """
        判断是否需要训练判别器
        
        策略：
        - Phase 1: 不训练
        - Phase 2/3: 每2个batch训练1次判别器
        """
        phase = self.get_current_phase(epoch)
        
        if phase == 1:
            return False
        
        # 每2个batch训练1次判别器
        return batch_idx % 2 == 0
    
    def get_freeze_config(self, epoch):
        """
        获取冻结配置
        
        Phase 1: 冻结CLIP后6层（保留预训练知识）
        Phase 2: 解冻所有CLIP层
        Phase 3: 保持解冻
        """
        phase = self.get_current_phase(epoch)
        
        if phase == 1:
            return {
                'clip_unfreeze_from_layer': 6,  # 只解冻前6层
                'vim_unfreeze_from_layer': 0,  # Vim全部解冻
                'freeze_bn': True  # 冻结BatchNorm
            }
        else:
            return {
                'clip_unfreeze_from_layer': 0,  # 全部解冻
                'vim_unfreeze_from_layer': 0,
                'freeze_bn': False
            }
    
    def print_phase_summary(self, epoch):
        """打印当前阶段摘要"""
        phase = self.get_current_phase(epoch)
        weights = self.get_loss_weights(epoch)
        lr_mult = self.get_learning_rate_multiplier(epoch)
        
        if self.logger:
            self.logger.logger.info("=" * 70)
            self.logger.logger.info(f"📚 Curriculum Learning - Phase {phase} (Epoch {epoch})")
            self.logger.logger.info("=" * 70)
            
            if phase == 1:
                self.logger.logger.info("🎯 Goal: Establish basic ID discrimination (Rank-1 > 30%)")
                self.logger.logger.info("🔧 Strategy: Disable decoupling, focus on Triplet Loss")
            elif phase == 2:
                self.logger.logger.info("🎯 Goal: Adversarial decoupling (Rank-1 > 50%)")
                self.logger.logger.info("🔧 Strategy: Gradually increase adversarial loss with GRL")
            else:
                self.logger.logger.info("🎯 Goal: Fine-tuning all modules (Rank-1 > 60%)")
                self.logger.logger.info("🔧 Strategy: Balance all losses, optimize fusion")
            
            self.logger.logger.info(f"📊 LR Multiplier: {lr_mult:.2f}x")
            self.logger.logger.info("📈 Active Loss Weights:")
            for key, val in weights.items():
                if val > 0:
                    self.logger.logger.info(f"  - {key}: {val:.4f}")
            self.logger.logger.info("=" * 70)
    
    def should_transition_phase(self, epoch, performance_history):
        """
        判断是否应该提前过渡到下一阶段
        
        条件：
        - Phase 1 -> Phase 2: Rank-1 > 30% 且 epoch >= 10
        - Phase 2 -> Phase 3: Rank-1 > 50% 且 epoch >= 30
        """
        if not performance_history or len(performance_history) == 0:
            return False
        
        phase = self.get_current_phase(epoch)
        latest_rank1 = performance_history[-1].get('rank1', 0.0)
        
        if phase == 1 and latest_rank1 > 0.30 and epoch >= 10:
            return True
        elif phase == 2 and latest_rank1 > 0.50 and epoch >= 30:
            return True
        
        return False
