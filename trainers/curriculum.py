# trainers/curriculum.py
import torch
import torch.nn as nn

class CurriculumScheduler:
    # 三阶段课程学习调度器：通过动态调整损失权重、学习率和模块冻结状态，实现从基础对齐到高级解耦的平滑训练
    # Phase 1 (Epoch 1-15): 基础 ID 判别训练
    # Phase 2 (Epoch 16-40): 对抗式特征解耦
    # Phase 3 (Epoch 41+): 全局精细化微调
    
    def __init__(self, total_epochs=80, logger=None):
        self.total_epochs = total_epochs
        self.logger = logger
        
        # 训练阶段界限设置
        self.phase1_end = 15
        self.phase2_end = 40
        
        # 初始阶段基础权重配置
        self.base_weights = {
            'info_nce': 1.0,           # 跨模态对比损失
            'id_triplet': 2.0,         # 身份三元组损失
            'cloth_semantic': 0.1,     # 服装语义损失
            'spatial_orthogonal': 0.0, # 空间正交损失（初期禁用）
            'semantic_alignment': 0.0, # 语义对齐损失（初期禁用）
            'ortho_reg': 0.0,          # 正交正则化项
            'adversarial_attr': 0.0,   # 属性对抗损失（初期禁用）
            'adversarial_domain': 0.0, # 域对抗损失（初期禁用）
            'discriminator_attr': 0.0, # 属性判别器损失
            'discriminator_domain': 0.0# 域判别器损失
        }
        
        if logger:
            logger.debug_logger.info("=" * 70)
            logger.debug_logger.info("📚 课程学习调度器已初始化")
            logger.debug_logger.info("=" * 70)
            logger.debug_logger.info(f"阶段 1 (Epoch 1-{self.phase1_end}): 基础 ID 判别训练")
            logger.debug_logger.info(f"阶段 2 (Epoch {self.phase1_end+1}-{self.phase2_end}): 对抗式特征解耦")
            logger.debug_logger.info(f"阶段 3 (Epoch {self.phase2_end+1}+): 全局精细化微调")
            logger.debug_logger.info("=" * 70)
    
    def get_current_phase(self, epoch):
        # 根据当前 Epoch 获取所属的训练阶段
        if epoch <= self.phase1_end:
            return 1
        elif epoch <= self.phase2_end:
            return 2
        else:
            return 3
    
    def get_loss_weights(self, epoch, performance_history=None):
        # 动态计算当前 Epoch 的各项损失权重，实现平滑过渡
        phase = self.get_current_phase(epoch)
        
        if phase == 1:
            # 第一阶段：侧重基础特征对齐和身份识别
            weights = {
                'info_nce': 1.0,
                'id_triplet': 2.0,
                'cloth_semantic': 0.1,
                'spatial_orthogonal': 0.0,
                'semantic_alignment': 0.0,
                'ortho_reg': 0.0,
                'adversarial_attr': 0.0,
                'adversarial_domain': 0.0,
                'discriminator_attr': 0.0,
                'discriminator_domain': 0.0
            }

            # 提前过渡检测：若 Rank-1 性能达标则提前开启解耦
            if performance_history and len(performance_history) > 0:
                latest_rank1 = performance_history[-1].get('rank1', 0.0)
                if latest_rank1 > 0.30 and epoch >= 10:
                    if self.logger:
                        self.logger.logger.info(f"🎯 性能触发提前过渡: Rank-1={latest_rank1:.1%} > 30%, 提前进入阶段 2")
                    return self.get_loss_weights(self.phase1_end + 1, performance_history)

        elif phase == 2:
            # 第二阶段：引入对抗解耦，并随进度线性增加解耦强度
            progress = (epoch - self.phase1_end) / (self.phase2_end - self.phase1_end)

            weights = {
                'info_nce': 1.0,
                'id_triplet': 2.0 - progress * 0.5,    # 降低 ID 损失占比
                'cloth_semantic': 0.1 + progress * 0.2, # 增加服装语义占比
                'spatial_orthogonal': progress * 0.3,   # 线性增加空间正交约束
                'semantic_alignment': progress * 0.05,  # 线性增加语义对齐约束
                'ortho_reg': progress * 0.2,           # 线性增加 Query 正交正则化
                'adversarial_attr': progress * 0.3,    # 线性增加属性对抗强度
                'adversarial_domain': progress * 0.1,  # 线性增加域对抗强度
                'discriminator_attr': 0.5,              # 固定判别器基础权重
                'discriminator_domain': 0.2
            }

            # 停滞检测：若性能增长乏力，则临时回拨解耦强度并加强 ID 学习
            if performance_history and len(performance_history) >= 5:
                recent_maps = [h.get('mAP', 0.0) for h in performance_history[-5:]]
                if max(recent_maps) - min(recent_maps) < 0.01:
                    if self.logger:
                        self.logger.logger.warning(f"⚠️ 检测到性能平台，动态调整权重以跳出局部最优")
                    weights['id_triplet'] *= 1.2
                    weights['adversarial_attr'] *= 0.5

        else:
            # 第三阶段：所有模块全速优化，保持解耦与性能的平衡
            weights = {
                'info_nce': 1.0,
                'id_triplet': 1.5,
                'cloth_semantic': 0.3,
                'spatial_orthogonal': 0.3,
                'semantic_alignment': 0.05,
                'ortho_reg': 0.2,
                'adversarial_attr': 0.2,
                'adversarial_domain': 0.05,
                'discriminator_attr': 0.3,
                'discriminator_domain': 0.1
            }
        
        return weights
    
    def get_learning_rate_multiplier(self, epoch):
        # 根据训练阶段动态缩放全局学习率
        phase = self.get_current_phase(epoch)
        if phase == 1:
            return 1.0 # 第一阶段全速对齐
        elif phase == 2:
            return 0.5 # 第二阶段半速解耦
        else:
            return 0.3 # 第三阶段低速微调
    
    def should_train_discriminator(self, epoch, batch_idx, total_batches):
        # 判别器训练频率调度：第一阶段不训练，后续阶段每两个 batch 训练一次
        phase = self.get_current_phase(epoch)
        if phase == 1:
            return False
        return batch_idx % 2 == 0
    
    def get_freeze_config(self, epoch):
        # 获取各阶段的模型冻结/解冻配置
        phase = self.get_current_phase(epoch)
        if phase == 1:
            return {
                'clip_unfreeze_from_layer': 6, # 仅解冻 CLIP 的深层
                'vim_unfreeze_from_layer': 0,  # 视觉编码器完全解冻
                'freeze_bn': True              # 冻结 BN 层以稳定初期统计量
            }
        else:
            return {
                'clip_unfreeze_from_layer': 0, # 全部解冻进行联合优化
                'vim_unfreeze_from_layer': 0,
                'freeze_bn': False
            }
    
    def print_phase_summary(self, epoch):
        # 格式化输出当前训练阶段的详细摘要
        import shutil
        phase = self.get_current_phase(epoch)
        weights = self.get_loss_weights(epoch)
        lr_mult = self.get_learning_rate_multiplier(epoch)
        
        term_width = shutil.get_terminal_size((80, 20)).columns
        width = min(max(term_width, 80), 100)
        
        phase_descriptions = {
            1: "骨干网络适配与特征对齐阶段",
            2: "基于对抗正则化的特征流形解耦阶段",
            3: "双流语义融合与全局精细微调阶段"
        }
        
        phase_strategies = {
            1: "带预热的标准 SGD | 解耦约束：已禁用",
            2: "梯度反转层 (GRL) | 对抗权重线性平滑爬升",
            3: "全模块联合优化 | 融合机制精细调优"
        }
        
        if self.logger:
            self.logger.logger.info(f"{'='*width}")
            title = f"🚀 课程学习调度 | Epoch {epoch} | 阶段 {phase}"
            self.logger.logger.info(f"{title}")
            self.logger.logger.info(f"{'-'*width}")
            self.logger.logger.info(f"  📌 阶段目标:        {phase_descriptions.get(phase, '未知阶段')}")
            self.logger.logger.info(f"  ⚙️  优化策略:        {phase_strategies.get(phase, '标准模式')}")
            self.logger.logger.info(f"  📉 学习率缩放:      {lr_mult:.4f}x")
            
            active_weights = [f"{k}={v:.4g}" for k, v in weights.items() if v > 1e-6]
            self.logger.logger.info(f"  ⚖️  动态损失权重:")
            for i in range(0, len(active_weights), 3):
                line = " | ".join(active_weights[i:i+3])
                self.logger.logger.info(f"      [{line}]")
            
            self.logger.logger.info(f"{'='*width}")
    
    def should_transition_phase(self, epoch, performance_history):
        # 逻辑判断：是否应当基于当前性能指标提前进入下一阶段
        if not performance_history or len(performance_history) == 0:
            return False
        
        phase = self.get_current_phase(epoch)
        latest_rank1 = performance_history[-1].get('rank1', 0.0)
        
        if phase == 1 and latest_rank1 > 0.30 and epoch >= 10:
            return True
        elif phase == 2 and latest_rank1 > 0.50 and epoch >= 30:
            return True
        
        return False
