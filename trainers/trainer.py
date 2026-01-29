# trainers/trainer.py
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
from losses.loss import Loss
from evaluators.evaluator import Evaluator
from utils.serialization import save_checkpoint
from utils.meters import AverageMeter
from utils.visualization import FSHDVisualizer
from trainers.curriculum import CurriculumScheduler

class EarlyStopping:
    # 早停机制类：当验证集性能在设定的耐心值（Patience）周期内未显著提升时，提前终止训练以防止过拟合
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
                    f"早停检查: {self.counter}/{self.patience} (最佳={self.best_score:.4f}, 当前={mAP:.4f})"
                )
            if self.counter >= self.patience:
                self.early_stop = True
                if self.logger:
                    self.logger.logger.warning("触发早停机制，训练提前结束")
        else:
            self.best_score = mAP
            self.counter = 0

class Trainer:
    # 训练器核心类：管理模型训练全生命周期，包括数据前向传播、损失计算、梯度更新、可视化及评估流程
    def __init__(self, model, args, monitor=None, runner=None):
        self.model = model
        self.args = args
        self.monitor = monitor
        self.runner = runner
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化课程学习调度器与性能历史
        self.curriculum = CurriculumScheduler(total_epochs=args.epochs, logger=monitor)
        self.performance_history = []

        # 配置损失计算模块（支持语义引导与对抗解耦）
        self.loss = Loss(
            temperature=0.07,
            weights=self.curriculum.base_weights,
            logger=monitor,
            semantic_guidance=model.semantic_guidance,
            adversarial_decoupler=model.adversarial_decoupler,
            base_lr=args.lr
        ).to(self.device)
        
        # 可视化器初始化
        visualize_config = getattr(args, 'visualization', {})
        if visualize_config.get('enabled', True):
            vis_save_dir = visualize_config.get('save_dir', 'visualizations')
            self.visualizer = FSHDVisualizer(save_dir=vis_save_dir, logger=monitor)
            self.visualize_freq = visualize_config.get('frequency', 5)
            self.visualize_batch_interval = visualize_config.get('batch_interval', 200)
            if self.monitor:
                self.monitor.debug_logger.info(f"✅ 可视化器已启动 (频率={self.visualize_freq} ep, 步长={self.visualize_batch_interval} batch)")
        else:
            self.visualizer = None
        
        # 混合精度训练配置
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16) if self.device.type == 'cuda' else None

    def reinit_clip_bias_layers(self, model, logger=None):
        # 针对 CLIP 文本编码器的 bias 层执行重新初始化，用于加速特定阶段的收敛
        reinitialized_count = 0
        for name, param in model.named_parameters():
            if 'text_encoder' in name and 'bias' in name and param.requires_grad:
                nn.init.normal_(param, std=0.02)
                reinitialized_count += 1
        if logger:
            logger.debug_logger.info(f"已重置 {reinitialized_count} 个 CLIP bias 参数")
    
    def build_optimizer_with_lr_groups(self, model, stage):
        # 为模型不同部分（如骨干网络与解耦模块）创建具有差异化学习率的优化器组
        if stage >= 2:
            clip_params, other_params = [], []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if 'text_encoder.text_model.encoder' in name:
                        try:
                            layer_num = int(name.split('.')[4])
                            if layer_num >= 11:
                                clip_params.append(param)
                                continue
                        except: pass
                    other_params.append(param)
            
            if clip_params:
                param_groups = [
                    {'params': clip_params, 'lr': self.args.lr * 0.5, 'name': 'clip_finetune'},
                    {'params': other_params, 'lr': self.args.lr, 'name': 'task_modules'}
                ]
                return torch.optim.AdamW(param_groups, weight_decay=self.args.weight_decay)
        return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=self.args.lr, weight_decay=self.args.weight_decay)
    
    def clip_grad_norm_by_layer(self, model, max_norm=1.0):
        # 实现分层梯度裁剪，针对 Vision Mamba 等数值敏感模块应用更严格的约束以防止梯度爆炸
        for name, param in model.named_parameters():
            if param.grad is not None:
                if 'visual_encoder' in name and 'mixer' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.3) # Mamba 核心层最严格
                elif 'visual_encoder' in name or 'text_encoder' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.5) # 骨干网络严格
                elif 'disentangle' in name or 'fusion' in name:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm * 0.7) # 任务模块适中
                else:
                    torch.nn.utils.clip_grad_norm_([param], max_norm=max_norm)       # 其他标准裁剪
    
    def run(self, inputs, epoch, batch_idx, total_batches, training_phase='feature'):
        # 执行前向传播并计算各组件损失，同时记录特征统计信息与可视化结果
        image, cloth_captions, id_captions, pid, cam_id, is_matched = inputs
        image = image.to(self.device)
        pid = pid.to(self.device)
        is_matched = is_matched.to(self.device)

        with torch.amp.autocast('cuda', enabled=self.args.fp16):
            outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)
            
            # 解包模型输出：涵盖特征向量、解耦辅助信息及注意力权重等
            image_embeds, id_text_embeds, fused_embeds, id_embeds, \
            cloth_embeds, cloth_text_embeds, cloth_image_embeds, aux_info, gate_weights, \
            id_cls_features, original_feat = outputs
            
        # 调用 Loss 模块计算聚合损失
        loss_dict = self.loss(
            image_embeds=image_embeds, id_text_embeds=id_text_embeds, fused_embeds=fused_embeds,
            id_logits=None, id_embeds=id_embeds, cloth_embeds=cloth_embeds,
            cloth_text_embeds=cloth_text_embeds, cloth_image_embeds=cloth_image_embeds,
            pids=pid, epoch=epoch, aux_info=aux_info, training_phase=training_phase
        )

        # 触发中间特征可视化
        if self.visualizer and epoch % self.visualize_freq == 0 and batch_idx % self.visualize_batch_interval == 0:
            if aux_info is not None and isinstance(aux_info, dict):
                self.visualizer.plot_attention_maps(aux_info, epoch, batch_idx, images=image)
        
        # 监控指标记录
        if self.monitor and batch_idx % 200 == 0:
            self.monitor.log_feature_statistics(image_embeds, "image_feat")
            self.monitor.log_feature_statistics(id_text_embeds, "text_feat")
            if aux_info is not None and isinstance(aux_info, dict):
                conflict_score = aux_info.get('conflict_score')
                if conflict_score is not None:
                    self.monitor.log_conflict_score(conflict_score, step_name=f"_E{epoch}_B{batch_idx}")
            if gate_weights is not None:
                self.monitor.log_gate_weights(gate_weights, "fusion_gate")
            self.monitor.log_loss_components(loss_dict)
            self.monitor.log_memory_usage()

        return loss_dict

    def train(self, train_loader, optimizer, lr_scheduler, query_loader=None, gallery_loader=None, checkpoint_dir=None):
        # 训练主入口：代理至课程学习训练循环
        from trainers.train_loop import train_with_curriculum
        return train_with_curriculum(
            trainer=self, train_loader=train_loader, query_loader=query_loader,
            gallery_loader=gallery_loader, checkpoint_dir=checkpoint_dir
        )

    def _get_dataset_name(self):
        # 内部辅助函数：解析数据集名称用于检查点命名
        if hasattr(self.args, 'dataset_configs') and self.args.dataset_configs:
            name = self.args.dataset_configs[0]['name'].lower()
            for k in ['cuhk', 'rstp', 'icfg']:
                if k in name: return k
            return name
        return 'unknown'
