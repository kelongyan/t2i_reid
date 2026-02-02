# trainers/curriculum.py
import torch
import torch.nn as nn

class CurriculumScheduler:
    """
    三阶段课程学习调度器 (优化版)
    
    模型结构适配策略：
    - Phase 1 (Epoch 1-20): 基础特征对齐
      * 重点：跨模态对比学习 (InfoNCE) + 身份判别 (Triplet)
      * 辅助：服装语义对齐 (为后续解耦铺垫)
      * 禁用：所有解耦相关损失
      
    - Phase 2 (Epoch 21-50): 渐进式特征解耦  
      * 引入：AH-Net空间正交约束 (Spatial Orthogonal)
      * 引入：语义引导对齐 (Semantic Alignment)
      * 引入：对抗解耦 (Adversarial) - 逐步增强
      * 降低：Triplet权重 (避免与解耦冲突)
      
    - Phase 3 (Epoch 51+): 全局精细微调
      * 降低：所有对抗损失 (稳定收敛)
      * 保持：空间正交 + 语义对齐 (维持解耦质量)
      * 微调：低学习率优化细节
    """
    
    def __init__(self, total_epochs=100, logger=None):
        self.total_epochs = total_epochs
        self.logger = logger
        
        # 调整阶段边界，给解耦更多时间
        self.phase1_end = 20
        self.phase2_end = 50
        
        # Phase 1: 基础对齐 (保守初始化)
        self.base_weights = {
            'info_nce': 1.0,           # 跨模态对比 (基准)
            'id_triplet': 5.0,         # 身份三元组 (适中，避免过大)
            'cloth_semantic': 1.0,     # 服装语义 (增强，为解耦铺垫)
            'spatial_orthogonal': 0.0, # 禁用
            'semantic_alignment': 0.0, # 禁用
            'ortho_reg': 0.0,          # 禁用
            'adversarial_attr': 0.0,   # 禁用
            'adversarial_domain': 0.0, # 禁用
            'discriminator_attr': 0.0, # 禁用
            'discriminator_domain': 0.0# 禁用
        }
        
        if logger:
            logger.logger.info("=" * 70)
            logger.logger.info("📚 Curriculum Scheduler Initialized (Optimized)")
            logger.logger.info("=" * 70)
            logger.logger.info(f"Phase 1 (Epoch 1-{self.phase1_end}): Base Alignment (InfoNCE + Triplet + Cloth)")
            logger.logger.info(f"Phase 2 (Epoch {self.phase1_end+1}-{self.phase2_end}): Progressive Disentanglement")
            logger.logger.info(f"Phase 3 (Epoch {self.phase2_end+1}+): Fine-tuning & Stabilization")
            logger.logger.info("=" * 70)
    
    def get_current_phase(self, epoch):
        if epoch <= self.phase1_end:
            return 1
        elif epoch <= self.phase2_end:
            return 2
        else:
            return 3
    
    def get_loss_weights(self, epoch, performance_history=None):
        phase = self.get_current_phase(epoch)
        
        if phase == 1:
            # Phase 1: 纯基础对齐
            weights = {
                'info_nce': 1.0,
                'id_triplet': 5.0,
                'cloth_semantic': 1.0,
                'spatial_orthogonal': 0.0,
                'semantic_alignment': 0.0,
                'ortho_reg': 0.0,
                'adversarial_attr': 0.0,
                'adversarial_domain': 0.0,
                'discriminator_attr': 0.0,
                'discriminator_domain': 0.0
            }
            
            # 早过渡检测：Rank-1 > 35% 可提前进入 Phase 2
            if performance_history and len(performance_history) > 0:
                latest_rank1 = performance_history[-1].get('rank1', 0.0)
                if latest_rank1 > 0.35 and epoch >= 15:
                    if self.logger:
                        self.logger.logger.info(f"🎯 Early transition triggered: Rank-1={latest_rank1:.1%}")
                    return self.get_loss_weights(self.phase1_end + 1, performance_history)
                    
        elif phase == 2:
            # Phase 2: 渐进式解耦 (线性增加解耦强度)
            progress = (epoch - self.phase1_end) / (self.phase2_end - self.phase1_end)  # 0~1
            
            weights = {
                # 基础损失 (随解耦增强而适度降低)
                'info_nce': 1.0,
                'id_triplet': max(3.0, 5.0 - progress * 1.5),  # 5.0 -> 3.5
                'cloth_semantic': 1.0 + progress * 0.5,        # 1.0 -> 1.5
                
                # AH-Net 空间解耦 (关键)
                'spatial_orthogonal': progress * 0.5,          # 0 -> 0.5
                'ortho_reg': progress * 0.3,                   # 0 -> 0.3
                
                # 语义引导对齐
                'semantic_alignment': progress * 0.2,          # 0 -> 0.2
                
                # 对抗解耦 (逐步增强，但控制上限避免不稳定)
                'adversarial_attr': min(0.5, progress * 0.6),  # 0 -> 0.5 (上限)
                'adversarial_domain': min(0.2, progress * 0.3),# 0 -> 0.2 (上限)
                'discriminator_attr': min(0.5, progress * 0.6),# 0 -> 0.5
                'discriminator_domain': min(0.2, progress * 0.3) # 0 -> 0.2
            }
            
            # 停滞检测：性能停滞时临时降低解耦强度
            if performance_history and len(performance_history) >= 5:
                recent_maps = [h.get('mAP', 0.0) for h in performance_history[-5:]]
                if max(recent_maps) - min(recent_maps) < 0.005:  # 5 epoch 无提升
                    if self.logger:
                        self.logger.logger.warning(f"⚠️ Plateau detected at epoch {epoch}, reducing disentanglement strength")
                    weights['adversarial_attr'] *= 0.5
                    weights['adversarial_domain'] *= 0.5
                    weights['spatial_orthogonal'] *= 0.7
                    
        else:
            # Phase 3: 精细微调 (降低对抗，保持解耦)
            weights = {
                'info_nce': 1.0,
                'id_triplet': 3.0,           # 稳定低值
                'cloth_semantic': 1.5,       # 保持
                'spatial_orthogonal': 0.5,   # 保持解耦质量
                'ortho_reg': 0.3,            # 保持
                'semantic_alignment': 0.2,   # 保持
                'adversarial_attr': 0.2,     # 降低对抗强度 (稳定)
                'adversarial_domain': 0.1,   # 降低
                'discriminator_attr': 0.2,   # 同步降低
                'discriminator_domain': 0.1  # 同步降低
            }
        
        return weights
    
    def get_learning_rate_multiplier(self, epoch):
        # 学习率衰减策略
        phase = self.get_current_phase(epoch)
        if phase == 1:
            return 1.0      # 全速
        elif phase == 2:
            return 0.7      # 中速 (解耦阶段降低LR提高稳定性)
        else:
            return 0.3      # 低速微调
    
    def should_train_discriminator(self, epoch, batch_idx, total_batches):
        # 判别器训练频率
        phase = self.get_current_phase(epoch)
        if phase == 1:
            return False
        # Phase 2-3: 每2个batch训练一次判别器 (与生成器交替)
        return batch_idx % 2 == 0
    
    def get_freeze_config(self, epoch):
        # 骨干网络解冻策略
        phase = self.get_current_phase(epoch)
        if phase == 1:
            return {
                'clip_unfreeze_from_layer': 8,  # 冻结前8层，仅训练深层
                'vim_unfreeze_from_layer': 0,   # Vim完全解冻 (任务适配)
                'freeze_bn': True               # 冻结BN统计量
            }
        elif phase == 2:
            return {
                'clip_unfreeze_from_layer': 4,  # 逐步解冻CLIP
                'vim_unfreeze_from_layer': 0,
                'freeze_bn': False              # 解冻BN
            }
        else:
            return {
                'clip_unfreeze_from_layer': 0,  # 全部解冻
                'vim_unfreeze_from_layer': 0,
                'freeze_bn': False
            }
    
    def print_phase_summary(self, epoch):
        import shutil
        phase = self.get_current_phase(epoch)
        weights = self.get_loss_weights(epoch)
        lr_mult = self.get_learning_rate_multiplier(epoch)
        
        term_width = shutil.get_terminal_size((80, 20)).columns
        width = min(max(term_width, 80), 100)
        
        phase_descriptions = {
            1: "🎯 Base Alignment: Cross-modal Feature Learning",
            2: "🔥 Progressive Disentanglement: AH-Net + Adversarial",
            3: "✨ Fine-tuning: Stabilization & Refinement"
        }
        
        if self.logger:
            self.logger.logger.info(f"{'='*width}")
            self.logger.logger.info(f"Curriculum Phase {phase} | Epoch {epoch}/{self.total_epochs}")
            self.logger.logger.info(f"{'-'*width}")
            self.logger.logger.info(f"  {phase_descriptions.get(phase, 'Unknown')}")
            self.logger.logger.info(f"  LR Multiplier: {lr_mult:.2f}x")
            
            # 显示活跃损失
            active = [(k, v) for k, v in weights.items() if v > 1e-6]
            active.sort(key=lambda x: -x[1])
            self.logger.logger.info(f"  Active Losses ({len(active)}):")
            for i in range(0, len(active), 2):
                line = " | ".join([f"{k}={v:.3f}" for k, v in active[i:i+2]])
                self.logger.logger.info(f"    {line}")
            
            self.logger.logger.info(f"{'='*width}")
    
    def should_transition_phase(self, epoch, performance_history):
        # 基于性能的智能阶段切换
        if not performance_history or len(performance_history) == 0:
            return False
        
        phase = self.get_current_phase(epoch)
        latest_rank1 = performance_history[-1].get('rank1', 0.0)
        
        # Phase 1 -> 2: Rank-1 > 35% 且训练了至少15 epoch
        if phase == 1 and latest_rank1 > 0.35 and epoch >= 15:
            return True
        # Phase 2 -> 3: Rank-1 > 55% 且训练了至少40 epoch
        elif phase == 2 and latest_rank1 > 0.55 and epoch >= 40:
            return True
        
        return False
