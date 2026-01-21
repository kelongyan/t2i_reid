# src/trainer/trainer.py
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from losses.loss import Loss
from evaluators.evaluator import Evaluator
from utils.serialization import save_checkpoint
from utils.meters import AverageMeter
from utils.visualization import FSHDVisualizer  # 新增：可视化工具

class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience=10, min_delta=0.001, logger=None):
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
        self.monitor = monitor  # 添加监控器
        self.runner = runner  # 添加runner引用以便调用freeze方法
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # === FSHD权重配置（激进版）===
        default_loss_weights = {
            'info_nce': 1.0, 
            'cls': 0.05,
            'cloth_semantic': 1.0, 
            'orthogonal': 0.1,            # 大幅降低
            'gate_adaptive': 0.02,
            'reconstruction': 0.5,
            'semantic_alignment': 0.1,     # 大幅降低
            'freq_consistency': 0.5,      # 【新增】频域一致性
            'freq_separation': 0.2,       # 【新增】频域分离
        }
        
        # 从配置文件获取损失权重，合并默认值
        loss_weights = getattr(args, 'disentangle', {}).get('loss_weights', default_loss_weights)
        for key, value in default_loss_weights.items():
            if key not in loss_weights:
                loss_weights[key] = value
        
        # 初始化Loss模块
        self.combined_loss = Loss(temperature=0.1, weights=loss_weights, logger=monitor).to(self.device)
        
        # === 设置语义引导模块到Loss（关键！）===
        if hasattr(model, 'semantic_guidance'):
            self.combined_loss.set_semantic_guidance(model.semantic_guidance)
            if self.monitor:
                self.monitor.debug_logger.info("✅ Semantic guidance module connected to Loss system")
        
        # === 新增：初始化可视化器 ===
        visualize_config = getattr(args, 'visualization', {})
        if visualize_config.get('enabled', True):
            vis_save_dir = visualize_config.get('save_dir', 'visualizations')
            self.visualizer = FSHDVisualizer(save_dir=vis_save_dir, logger=monitor)
            self.visualize_freq = visualize_config.get('frequency', 5)  # 每N个epoch可视化一次
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
        """对不同层使用不同的梯度裁剪阈值"""
        for name, param in model.named_parameters():
            if param.grad is not None:
                # CLIP文本编码器：更严格的裁剪
                if 'text_encoder' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.5)
                # 新解冻的层：较宽松的裁剪
                elif 'layers' in name:
                    try:
                        layer_num = int([s for s in name.split('.') if s.isdigit()][0])
                        if layer_num >= 11:
                            torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 2.0)
                        else:
                            torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm)
                    except (IndexError, ValueError):
                        torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm)
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

    def run(self, inputs, epoch, batch_idx, total_batches):
        # 执行单次训练步骤，计算所有损失（FSHD版本）
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
            # === FSHD模块支持返回频域信息 ===
            # 如果使用FSHD模块，需要获取频域信息
            return_freq_info = (self.visualizer is not None and 
                               batch_idx % self.visualize_batch_interval == 0)
            
            # 训练时可以选择性返回注意力图和频域信息
            outputs = self.model(image=image, cloth_instruction=cloth_captions, 
                               id_instruction=id_captions)

            # === FSHD模块返回12个输出（保持兼容）===
            if len(outputs) != 12:
                raise ValueError(f"Expected 12 model outputs during training, got {len(outputs)}")

            image_feats, id_text_feats, fused_feats, id_logits, id_embeds, \
            cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate_stats, gate_weights, \
            id_cls_features, original_feat = outputs
            
            # === 获取频域信息（如果使用FSHD模块）===
            freq_info = None
            if hasattr(self.model, 'disentangle') and hasattr(self.model.disentangle, 'forward'):
                # 检查是否是FSHD模块（通过检查是否有freq_splitter属性）
                if hasattr(self.model.disentangle, 'freq_splitter') and return_freq_info:
                    # 重新调用disentangle获取频域信息（仅用于可视化，不参与梯度）
                    with torch.no_grad():
                        # 从模型中提取image_embeds_raw
                        if self.model.vision_backbone_type == 'vim':
                            image_embeds_raw = self.model.visual_encoder(image)
                        else:
                            image_outputs = self.model.visual_encoder(image)
                            image_embeds_raw = image_outputs.last_hidden_state
                        image_embeds_raw = self.model.visual_proj(image_embeds_raw)
                        
                        # 调用disentangle获取freq_info
                        _, _, _, _, freq_info = self.model.disentangle(
                            image_embeds_raw, return_freq_info=True
                        )
            
            # === 损失计算（新增freq_info参数）===
            loss_dict = self.combined_loss(
                image_embeds=image_feats, id_text_embeds=id_text_feats, fused_embeds=fused_feats,
                id_logits=id_logits, id_embeds=id_embeds, cloth_embeds=cloth_embeds,
                cloth_text_embeds=cloth_text_embeds, cloth_image_embeds=cloth_image_embeds,
                pids=pid, is_matched=is_matched, epoch=epoch, gate=gate_stats,
                id_cls_features=id_cls_features, original_feat=original_feat,
                freq_info=freq_info  # 【新增】传递频域信息
            )

        # === 可视化回调 ===
        if self.visualizer is not None and batch_idx % self.visualize_batch_interval == 0:
            # 频域掩码可视化
            if freq_info is not None:
                self.visualizer.plot_frequency_masks(freq_info, epoch, batch_idx)
                
                # 频域能量谱
                if 'freq_magnitude' in freq_info:
                    self.visualizer.plot_frequency_energy_spectrum(freq_info, epoch, batch_idx)
            
            # 门控统计
            if gate_stats is not None and isinstance(gate_stats, dict):
                # 从gate_stats中提取实际的gate tensor（如果存在）
                # 注意：当前gate_stats只包含统计值，如果需要可视化需要修改模型返回gate tensor
                pass
        
        # 记录模型内部状态信息
        if self.monitor and batch_idx % 200 == 0:  # 每200个批次记录一次详细信息
            self.monitor.log_feature_statistics(image_feats, "image_features")
            self.monitor.log_feature_statistics(id_text_feats, "id_text_features")
            self.monitor.log_feature_statistics(fused_feats, "fused_features")
            self.monitor.log_feature_statistics(id_embeds, "identity_embeds")
            self.monitor.log_feature_statistics(cloth_embeds, "clothing_embeds")
            self.monitor.log_feature_statistics(cloth_text_embeds, "cloth_text_embeds")
            self.monitor.log_feature_statistics(cloth_image_embeds, "cloth_image_embeds")

            # gate_stats是dict，记录统计信息
            if gate_stats is not None and isinstance(gate_stats, dict):
                self.monitor.debug_logger.debug(
                    f"Gate stats: ID[{gate_stats.get('gate_id_mean', 0):.4f}], "
                    f"Attr[{gate_stats.get('gate_attr_mean', 0):.4f}], "
                    f"Diversity[{gate_stats.get('diversity', 0):.4f}]"
                )
                
                # 【新增】频域信息记录
                if 'freq_type' in gate_stats:
                    self.monitor.debug_logger.debug(
                        f"Frequency: type={gate_stats.get('freq_type')}, "
                        f"energy={gate_stats.get('low_freq_energy', 0):.4f}"
                    )
            
            if gate_weights is not None:
                self.monitor.log_gate_weights(gate_weights, "fusion_gate")

            self.monitor.log_loss_components(loss_dict)

            # 记录内存使用情况
            self.monitor.log_memory_usage()

        return loss_dict

    def compute_similarity(self, train_loader):
        # 计算图像和文本特征的相似度
        self.model.eval()
        with torch.no_grad():
            for image, cloth_captions, id_captions, pid, cam_id, is_matched in train_loader:
                image = image.to(self.device)
                outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)
                # 对称解耦：12个输出
                image_feats, id_text_feats, _, _, _, _, _, _, gate_weights, _, _, _ = outputs
                sim = torch.matmul(image_feats, id_text_feats.t())
                pos_sim = sim.diag().mean().item()
                neg_sim = sim[~torch.eye(sim.shape[0], dtype=bool, device=self.device)].mean().item()
                scale = self.model.scale
                return pos_sim, neg_sim, None, scale
        self.model.train()
        return None, None, None, None

    def _format_loss_display(self, loss_meters):
        # 格式化损失显示，按指定顺序排列并隐藏特定项
        display_order = ['info_nce', 'cls', 'cloth_semantic', 'id_triplet', 'anti_collapse', 'gate_adaptive', 'reconstruction', 'total']
        hidden_losses = set()  # 所有损失都显示

        avg_losses = []
        for key in display_order:
            if key in loss_meters and loss_meters[key].count > 0:
                avg_losses.append(f"{key}={loss_meters[key].avg:.4f}")

        return avg_losses

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, gallery_loader=None, checkpoint_dir=None):
        # 训练模型，包含损失计算、优化和检查点保存
        self.model.train()
        best_mAP = 0.0
        best_checkpoint_path = None
        total_batches = len(train_loader)
        loss_meters = {k: AverageMeter() for k in self.combined_loss.weights.keys() | {'total'}}
        
        # 【新增】早停机制
        early_stopping = EarlyStopping(patience=10, min_delta=0.001, logger=self.monitor)
        
        # 【新增】学习率预热和全局步数
        warmup_steps = 1000
        global_step = 0

        for epoch in range(1, self.args.epochs + 1):
            # 【方案B：渐进解冻策略】在特定epoch检查并调整冻结状态和优化器
            stage_changed = False
            if self.runner:
                if epoch == 11:  # Stage 2: Vim后8层 + CLIP后1层
                    print("\n" + "="*70)
                    if self.monitor: self.monitor.logger.info("🔓 Progressive Unfreezing: Stage 2")
                    if self.monitor: self.monitor.logger.info("=" * 70)
                    if self.monitor: self.monitor.logger.info("Epoch 11-30: Unfreezing Vim last 8 layers (layer 16-23)")
                    if self.monitor: self.monitor.logger.info("             + CLIP last 1 layer (layer 11)")
                    if self.monitor: self.monitor.logger.info("Goal: Initial adaptation of CLIP semantic space")
                    print("="*70 + "\n")
                    self.runner.freeze_text_layers(self.model, unfreeze_from_layer=11)
                    self.runner.freeze_vit_layers(self.model, unfreeze_from_layer=4)
                    
                    # 【新增】重新初始化CLIP bias防止梯度消失
                    self.reinit_clip_bias_layers(self.model, self.monitor)
                    
                    # 【新增】使用分层学习率优化器
                    optimizer = self.build_optimizer_with_lr_groups(self.model, stage=2)
                    lr_scheduler = self.build_scheduler_with_cosine_warmup(
                        optimizer, 
                        num_training_steps=(self.args.epochs - 10) * total_batches,
                        num_warmup_steps=warmup_steps
                    )
                    
                    # 【新增】启用BatchNorm预热
                    self.enable_batch_norm_warmup(self.model, momentum=0.01)
                    
                    stage_changed = True
                    global_step = 0  # 重置全局步数
                elif epoch == 31:  # Stage 3: Vim后12层 + CLIP后6层
                    print("\n" + "="*70)
                    if self.monitor: self.monitor.logger.info("🔓 Progressive Unfreezing: Stage 3")
                    if self.monitor: self.monitor.logger.info("=" * 70)
                    if self.monitor: self.monitor.logger.info("Epoch 31-60: Unfreezing Vim last 12 layers")
                    if self.monitor: self.monitor.logger.info("             + CLIP last 6 layers (layer 6-11)")
                    if self.monitor: self.monitor.logger.info("Goal: Deep interaction tuning")
                    print("="*70 + "\n")
                    self.runner.freeze_text_layers(self.model, unfreeze_from_layer=6)
                    self.runner.freeze_vit_layers(self.model, unfreeze_from_layer=6)
                    
                    # 【新增】使用分层学习率优化器
                    optimizer = self.build_optimizer_with_lr_groups(self.model, stage=3)
                    lr_scheduler = self.build_scheduler_with_cosine_warmup(
                        optimizer,
                        num_training_steps=(self.args.epochs - 30) * total_batches,
                        num_warmup_steps=warmup_steps
                    )
                    
                    # 【新增】启用BatchNorm预热
                    self.enable_batch_norm_warmup(self.model, momentum=0.01)
                    
                    stage_changed = True
                    global_step = 0  # 重置全局步数
                elif epoch == 61:  # Stage 4: 全部解冻
                    print("\n" + "="*70)
                    if self.monitor: self.monitor.logger.info("🔓 Progressive Unfreezing: Stage 4")
                    if self.monitor: self.monitor.logger.info("=" * 70)
                    if self.monitor: self.monitor.logger.info("Epoch 61-80: Unfreezing all CLIP and Vim layers")
                    if self.monitor: self.monitor.logger.info("Goal: End-to-end fine-tuning")
                    print("="*70 + "\n")
                    self.runner.freeze_text_layers(self.model, unfreeze_from_layer=0)
                    self.runner.freeze_vit_layers(self.model, unfreeze_from_layer=0)
                    
                    # 【新增】使用默认优化器（所有层相同学习率）
                    optimizer = self._build_default_optimizer(self.model)
                    lr_scheduler = self.build_scheduler_with_cosine_warmup(
                        optimizer,
                        num_training_steps=(self.args.epochs - 60) * total_batches,
                        num_warmup_steps=warmup_steps
                    )
                    
                    # 【新增】启用BatchNorm预热
                    self.enable_batch_norm_warmup(self.model, momentum=0.01)
                    
                    stage_changed = True
                    global_step = 0  # 重置全局步数
            
            if stage_changed and self.monitor:
                self.monitor.logger.info(f"Stage changed at epoch {epoch}")
                if self.monitor:
                    self.monitor.logger.info(f"Learning rate warmup enabled for {warmup_steps} steps")
            
            # 显示上一个epoch的平均损失（仅记录到日志，不在终端显示以避免重复）
            if epoch > 1:
                avg_losses = self._format_loss_display(loss_meters)
                if avg_losses:
                    avg_loss_str = ', '.join(avg_losses)
                    # 仅记录到日志，评估阶段会单独打印损失
                    if self.monitor:
                        self.monitor.logger.info(f"[Epoch {epoch-1} Avg Loss]: {avg_loss_str}")

            # 重置损失记录器
            for meter in loss_meters.values():
                meter.reset()

            progress_bar = tqdm(
                train_loader, desc=f"[Epoch {epoch}/{self.args.epochs}] Training",
                dynamic_ncols=True, leave=True, total=total_batches
            )

            # 记录Epoch初始状态 (LR & Loss Weights) -> 仅写入调试日志，不显示在终端
            if self.monitor:
                current_lrs = [pg['lr'] for pg in optimizer.param_groups]
                lr_str = ", ".join([f"{lr:.2e}" for lr in current_lrs])
                
                # 获取当前Loss权重
                weight_str = ", ".join([f"{k}={v:.2f}" for k, v in self.combined_loss.weights.items() if v > 0])
                
                self.monitor.debug_logger.info(f"Epoch {epoch} Start | LRs: [{lr_str}] | Active Weights: [{weight_str}]")

            for i, inputs in enumerate(progress_bar):
                # 【新增】学习率预热
                if stage_changed and global_step < warmup_steps:
                    for param_group in optimizer.param_groups:
                        base_lr = param_group.get('initial_lr', param_group['lr'])
                        warmup_lr = self._get_warmup_lr(base_lr, global_step, warmup_steps)
                        param_group['lr'] = warmup_lr
                
                optimizer.zero_grad()
                loss_dict = self.run(inputs, epoch, i, total_batches)
                loss = loss_dict['total']

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # [Fix] Check for gradients BEFORE unscale to prevent scaler errors
                    has_grads = any(p.grad is not None for group in optimizer.param_groups for p in group['params'])
                    
                    if has_grads:
                        # 【修改】使用分层梯度裁剪
                        self.scaler.unscale_(optimizer)
                        self.clip_grad_norm_by_layer(self.model, max_norm=5.0)

                        # 记录梯度信息（每100个batch）
                        if self.monitor and i % 100 == 0:
                            # log_gradients 现在包含了原来的 flow analysis 功能
                            self.monitor.log_gradients(self.model, f"epoch_{epoch}_batch_{i}")

                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        if self.monitor: self.monitor.debug_logger.warning(f"⚠️  Skipping step at epoch {epoch} batch {i}: No gradients found (likely disconnected graph or unused params).")
                        # Debug info for first occurrence
                        if i == 0:
                            trainable_params = [n for n, p in self.model.named_parameters() if p.requires_grad]
                            if self.monitor: self.monitor.debug_logger.warning(f"Trainable params count: {len(trainable_params)}")
                            if self.monitor: self.monitor.debug_logger.warning("Sample trainable params with None grad:")
                            count = 0
                            for n, p in self.model.named_parameters():
                                if p.requires_grad and p.grad is None:
                                    if self.monitor: self.monitor.debug_logger.warning(f"  - {n}")
                                    count += 1
                                    if count > 10: break
                else:
                    loss.backward()
                    
                    # 【修改】使用分层梯度裁剪
                    self.clip_grad_norm_by_layer(self.model, max_norm=5.0)

                    # 记录梯度信息（每100个batch）
                    if self.monitor and i % 100 == 0:
                        self.monitor.log_gradients(self.model, f"epoch_{epoch}_batch_{i}")

                    optimizer.step()

                # 更新损失记录
                for key, val in loss_dict.items():
                    if key in loss_meters:
                        loss_meters[key].update(val.item() if isinstance(val, torch.Tensor) else val)
                
                # 记录详细损失分解（每100个batch）
                if self.monitor and i % 100 == 0:
                    self.monitor.log_loss_breakdown(loss_dict, epoch, i)

                # 记录批次信息
                if self.monitor and i % 200 == 0:  # 每200个批次记录一次
                    current_lr = optimizer.param_groups[0]['lr']
                    self.monitor.log_batch_info(epoch, i, total_batches,
                                              {k: v.avg for k, v in loss_meters.items()},
                                              current_lr, print_to_console=False)
                
                global_step += 1

            progress_bar.close()
            
            # 只在stage未改变时调用lr_scheduler.step()
            if not stage_changed:
                lr_scheduler.step()

            # 记录epoch信息
            if self.monitor:
                epoch_metrics = {k: v.avg for k, v in loss_meters.items()}
                self.monitor.log_epoch_info(epoch, self.args.epochs, epoch_metrics)

            # === 清理显存，准备评估 ===
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 每个epoch结束后进行评估
            if query_loader and gallery_loader:
                # 评估模型
                evaluator = Evaluator(self.model, args=self.args)
                metrics = evaluator.evaluate(
                    query_loader, gallery_loader, query_loader.dataset.data,
                    gallery_loader.dataset.data, checkpoint_path=None, epoch=epoch
                )

                current_mAP = metrics['mAP']

                # 同时在终端和日志显示评估结果
                print(f"\n{'='*60}")
                print(f"Epoch {epoch} Evaluation Results:")
                print(f"  mAP:    {metrics['mAP']:.4f}")
                print(f"  Rank-1: {metrics['rank1']:.4f}")
                print(f"  Rank-5: {metrics['rank5']:.4f}")
                print(f"  Rank-10: {metrics['rank10']:.4f}")
                print(f"{'='*60}\n")
                
                # 同时记录到日志文件
                if self.monitor:
                    self.monitor.logger.info(f"Epoch {epoch}: mAP={metrics['mAP']:.4f}, R1={metrics['rank1']:.4f}, R5={metrics['rank5']:.4f}, R10={metrics['rank10']:.4f}")

                # 【新增】早停检查
                early_stopping(current_mAP)
                if early_stopping.early_stop:
                    if self.monitor:
                        self.monitor.logger.info(f"Training stopped early at epoch {epoch}")
                    break

                # 保存最优检查点
                if current_mAP > best_mAP:
                    best_mAP = current_mAP

                    # 生成最佳检查点路径
                    if checkpoint_dir:
                        # 确保 checkpoint_dir 是 Path 对象
                        ckpt_dir_path = Path(checkpoint_dir)
                        
                        # 创建 model 子目录
                        model_dir = ckpt_dir_path / 'model'
                        model_dir.mkdir(parents=True, exist_ok=True)

                        # 获取数据集短名称用于文件名 (例如 cuhk, rstp, icfg)
                        dataset_short_name = self._get_dataset_name()
                        
                        # 构建完整路径: log/dataset_name/model/best_dataset.pth
                        new_best_checkpoint_path = str(model_dir / f"best_{dataset_short_name}.pth")

                        # 删除旧的最佳检查点
                        if best_checkpoint_path and Path(best_checkpoint_path).exists():
                            try:
                                Path(best_checkpoint_path).unlink()
                                if self.monitor:
                                    self.monitor.logger.info(f"Removed old best checkpoint: {best_checkpoint_path}")
                            except OSError:
                                if self.monitor:
                                    self.monitor.logger.warning(f"Could not remove old best checkpoint: {best_checkpoint_path}")

                        # 保存新的最佳检查点
                        save_checkpoint({
                            'model': self.model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'mAP': current_mAP
                        }, fpath=new_best_checkpoint_path)

                        best_checkpoint_path = new_best_checkpoint_path

                        if self.monitor:
                            self.monitor.debug_logger.debug(f"New best checkpoint saved: {best_checkpoint_path}, mAP: {best_mAP:.4f}")
                    else:
                        if self.monitor:
                            self.monitor.logger.warning("checkpoint_dir not provided, cannot save best checkpoint")

        # 显示训练完成信息（终端+日志）
        print(f"\n{'='*60}")
        print(f"🎉 Training Completed!")
        print(f"   Best mAP: {best_mAP:.4f}")
        if best_checkpoint_path:
            print(f"   Best Model: {best_checkpoint_path}")
        print(f"{'='*60}\n")
        
        if self.monitor:
            self.monitor.logger.info(f"Training completed. Best mAP: {best_mAP:.4f}")

        # 显示最终平均损失
        avg_losses = self._format_loss_display(loss_meters)
        if avg_losses:
            avg_loss_str = ', '.join(avg_losses)
            print(f"[Final Avg Loss]: {avg_loss_str}")
            if self.monitor:
                self.monitor.logger.info(f"[Final Avg Loss]: {avg_loss_str}")

        if best_checkpoint_path:
            if self.monitor: self.monitor.logger.info(f"Final best checkpoint: {best_checkpoint_path}, mAP: {best_mAP:.4f}")

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
