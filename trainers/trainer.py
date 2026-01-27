# src/trainer/trainer.py
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from losses.loss import Loss
from evaluators.evaluator import Evaluator
from utils.serialization import save_checkpoint
from utils.meters import AverageMeter
from utils.visualization import FSHDVisualizer
from trainers.curriculum import CurriculumScheduler  # 🔥 新增

class EarlyStopping:
    """早停机制，防止过拟合（修改为20个epoch）"""
    def __init__(self, patience=20, min_delta=0.001, logger=None):
        self.patience = patience
        self.min_delta = min_delta
        self.logger = logger
        self.best_score = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, mAP):
        if self.best_score is None:
            self.best_score = mAP
        elif mAP < self.best_score - self.min_delta:
            self.counter += 1
            if self.logger:
                self.logger.debug_logger.info(
                    f"EarlyStopping: {self.counter}/{self.patience} "
                    f"(best={self.best_score:.4f}, current={mAP:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.logger:
                    self.logger.logger.warning("Early stopping triggered!")
        else:
            self.best_score = mAP
            self.counter = 0

class Trainer:
    def __init__(self, model, args, monitor=None, runner=None):
        # 初始化训练器，设置模型、参数和设备
        self.model = model
        self.args = args
        self.monitor = monitor
        self.runner = runner
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 🔥 新增：课程学习调度器
        self.curriculum = CurriculumScheduler(
            total_epochs=args.epochs,
            logger=monitor
        )

        # 性能历史记录（用于课程学习）
        self.performance_history = []

        # 🔥 修复：初始化Loss模块（支持对抗训练，使用正确的温度参数）
        self.loss = Loss(
            temperature=0.07,  # 标准的InfoNCE温度参数
            weights=self.curriculum.base_weights,  # 使用课程学习的初始权重
            logger=monitor,
            semantic_guidance=model.semantic_guidance,
            adversarial_decoupler=model.adversarial_decoupler  # 🔥 新增
        ).to(self.device)
        
        # === 初始化可视化器 ===
        visualize_config = getattr(args, 'visualization', {})
        if visualize_config.get('enabled', True):
            vis_save_dir = visualize_config.get('save_dir', 'visualizations')
            self.visualizer = FSHDVisualizer(save_dir=vis_save_dir, logger=monitor)
            self.visualize_freq = visualize_config.get('frequency', 5)
            self.visualize_batch_interval = visualize_config.get('batch_interval', 200)
            if self.monitor:
                self.monitor.debug_logger.info(f"✅ Visualizer enabled (freq={self.visualize_freq}, batch_interval={self.visualize_batch_interval})")
        else:
            self.visualizer = None
        
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16) if self.device.type == 'cuda' else None
        if args.fp16 and self.device.type != 'cuda':
            if self.monitor: 
                self.monitor.logger.warning("FP16 is enabled but no CUDA device is available. Disabling mixed precision.")

    def reinit_clip_bias_layers(self, model, logger=None):
        """重新初始化CLIP文本编码器的bias，防止梯度消失"""
        reinitialized_count = 0
        for name, param in model.named_parameters():
            if 'text_encoder' in name and 'bias' in name and param.requires_grad:
                # 使用较小的std初始化
                nn.init.normal_(param, std=0.02)
                reinitialized_count += 1
                if logger and reinitialized_count <= 5:  # 只打印前5个
                    logger.debug_logger.info(f"Reinitialized CLIP bias: {name}")
        if logger:
            logger.debug_logger.info(f"Total CLIP bias params reinitialized: {reinitialized_count}")
    
    def build_optimizer_with_lr_groups(self, model, stage):
        """为新解冻层设置独立学习率"""
        if stage >= 2:
            # CLIP文本编码器后几层使用0.5倍学习率
            clip_params = []
            other_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'text_encoder.text_model.encoder' in name:
                        try:
                            layer_num = int(name.split('.')[4])  # text_model.encoder.layers.11
                            if layer_num >= 11:
                                clip_params.append(param)
                                continue
                        except (IndexError, ValueError):
                            pass
                    other_params.append(param)
            
            if clip_params:
                param_groups = [
                    {'params': clip_params, 'lr': self.args.lr * 0.5, 'name': 'clip_text', 'weight_decay': self.args.weight_decay},
                    {'params': other_params, 'lr': self.args.lr, 'name': 'others', 'weight_decay': self.args.weight_decay}
                ]
                if self.monitor:
                    self.monitor.logger.info(f"Built optimizer with {len(clip_params)} CLIP params (0.5x lr) and {len(other_params)} other params")
                return torch.optim.AdamW(param_groups)
        return self._build_default_optimizer(model)
    
    def _build_default_optimizer(self, model):
        """默认优化器构建方法"""
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
    
    def _get_warmup_lr(self, base_lr, current_step, warmup_steps):
        """学习率预热"""
        if current_step < warmup_steps:
            return base_lr * (current_step / warmup_steps)
        return base_lr
    
    def build_scheduler_with_cosine_warmup(self, optimizer, num_training_steps, num_warmup_steps):
        """余弦退火+预热学习率"""
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_training_steps - num_warmup_steps
        )
    
    def clip_grad_norm_by_layer(self, model, max_norm=1.0):
        """🔥 改进的分层梯度裁剪，特别针对Mamba模块"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                # 🔥 Vim Mamba模块：最严格的裁剪（防止NaN）
                if 'visual_encoder' in name and 'mixer' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.3)
                # Vim其他层：中等裁剪
                elif 'visual_encoder' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.5)
                # CLIP文本编码器：严格裁剪
                elif 'text_encoder' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.5)
                # 解耦模块：宽松裁剪
                elif 'disentangle' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.7)
                # Fusion模块：宽松裁剪
                elif 'fusion' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.8)
                # 其他层：标准裁剪
                else:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm)
    
    def enable_batch_norm_warmup(self, model, momentum=0.01):
        """为新解冻的层启用BatchNorm预热"""
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                module.momentum = momentum  # 降低momentum，加快统计量更新
                module.track_running_stats = True
        if self.monitor:
            self.monitor.logger.info(f"BatchNorm warmup enabled with momentum={momentum}")

    def run(self, inputs, epoch, batch_idx, total_batches, training_phase='feature'):
        # 执行单次训练步骤，计算所有损失（支持对抗训练）
        image, cloth_captions, id_captions, pid, cam_id, is_matched = inputs
        image = image.to(self.device)
        pid = pid.to(self.device)
        cam_id = cam_id.to(self.device) if cam_id is not None else None
        is_matched = is_matched.to(self.device)

        # 验证输入格式
        if batch_idx == 0:
            if not isinstance(cloth_captions, (list, tuple)) or not all(isinstance(c, str) for c in cloth_captions):
                raise ValueError("cloth_captions must be a list of strings")
            if not isinstance(id_captions, (list, tuple)) or not all(isinstance(c, str) for c in id_captions):
                raise ValueError("id_captions must be a list of strings")

        with torch.amp.autocast('cuda', enabled=self.args.fp16):
            # 训练时可以选择性返回注意力图
            outputs = self.model(image=image, cloth_instruction=cloth_captions, 
                               id_instruction=id_captions)

            # 模型返回11个输出
            if len(outputs) != 11:
                raise ValueError(f"Expected 11 model outputs during training, got {len(outputs)}")

            image_embeds, id_text_embeds, fused_embeds, id_embeds, \
            cloth_embeds, cloth_text_embeds, cloth_image_embeds, aux_info, gate_weights, \
            id_cls_features, original_feat = outputs
            
        # 🔥 计算损失（支持训练阶段区分）
        loss_dict = self.loss(
            image_embeds=image_embeds,
            id_text_embeds=id_text_embeds,
            fused_embeds=fused_embeds,
            id_logits=None,
            id_embeds=id_embeds,
            cloth_embeds=cloth_embeds,
            cloth_text_embeds=cloth_text_embeds,
            cloth_image_embeds=cloth_image_embeds,
            pids=pid,
            epoch=epoch,
            aux_info=aux_info,
            training_phase=training_phase  # 🔥 新增：区分特征提取器/判别器训练
        )

        # 可视化
        if self.visualizer and epoch % self.visualize_freq == 0 and batch_idx % self.visualize_batch_interval == 0:
            if hasattr(self.model.disentangle, 'forward'):
                if len(outputs) > 8:
                    aux_info = outputs[8]
                    if aux_info is not None and isinstance(aux_info, dict):
                        self.visualizer.plot_attention_maps(aux_info, epoch, batch_idx, images=image)
        
        # 记录特征统计信息
        if self.monitor and batch_idx % 200 == 0:
            self.monitor.log_feature_statistics(image_embeds, "image_features")
            self.monitor.log_feature_statistics(id_text_embeds, "id_text_features")
            self.monitor.log_feature_statistics(fused_embeds, "fused_features")
            self.monitor.log_feature_statistics(id_embeds, "identity_embeds")
            self.monitor.log_feature_statistics(cloth_embeds, "clothing_embeds")
            self.monitor.log_feature_statistics(cloth_text_embeds, "cloth_text_embeds")
            self.monitor.log_feature_statistics(cloth_image_embeds, "cloth_image_embeds")
            
            # Conflict Score日志
            if aux_info is not None and isinstance(aux_info, dict):
                conflict_score = aux_info.get('conflict_score')
                if conflict_score is not None and self.monitor:
                    if batch_idx % 200 == 0:
                        self.monitor.log_conflict_score(conflict_score, step_name=f"_E{epoch}_B{batch_idx}")

        # aux_info统计
        if aux_info is not None and isinstance(aux_info, dict):
            conflict_score = aux_info.get('conflict_score')
            ortho_reg = aux_info.get('ortho_reg')
            
            # 🔥 修复: 将Tensor转换为标量值用于日志输出
            conflict_val = conflict_score.mean().item() if conflict_score is not None else 0.0
            ortho_val = ortho_reg.item() if ortho_reg is not None else 0.0
            
            self.monitor.debug_logger.debug(
                f"Aux info: Conflict[{conflict_val:.4f}], "
                f"Ortho Reg[{ortho_val:.4f}]"
            )

        if gate_weights is not None:
            self.monitor.log_gate_weights(gate_weights, "fusion_gate")

        self.monitor.log_loss_components(loss_dict)

        # 记录内存使用情况
        self.monitor.log_memory_usage()

        return loss_dict
    

    def _format_loss_display(self, loss_meters):
        # 格式化损失显示，按指定顺序排列并隐藏特定项
        # [Modify] 适配 AH-Net + 方案书 Phase 3
        display_order = ['info_nce', 'reconstruction', 'cloth_semantic', 'id_triplet',
                        'spatial_orthogonal', 'semantic_alignment', 'total']

        avg_losses = []
        for key in display_order:
            if key in loss_meters and loss_meters[key].count > 0:
                avg_losses.append(f"{key}={loss_meters[key].avg:.4f}")

        return avg_losses

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, gallery_loader=None, checkpoint_dir=None):
        """训练模型（使用课程学习）"""
        from trainers.train_loop import train_with_curriculum
        
        # 调用新的训练循环
        best_mAP, best_checkpoint_path = train_with_curriculum(
            trainer=self,
            train_loader=train_loader,
            query_loader=query_loader,
            gallery_loader=gallery_loader,
            checkpoint_dir=checkpoint_dir
        )
        
        return best_mAP, best_checkpoint_path

    def _get_dataset_name(self):
        """获取数据集名称用于模型文件命名"""
        if hasattr(self.args, 'dataset_configs') and self.args.dataset_configs:
            dataset_name = self.args.dataset_configs[0]['name'].lower()
            if 'cuhk' in dataset_name:
                return 'cuhk'
            elif 'rstp' in dataset_name:
                return 'rstp'
            elif 'icfg' in dataset_name:
                return 'icfg'
            else:
                return dataset_name
        else:
            return 'unknown'