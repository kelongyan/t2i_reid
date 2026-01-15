# src/trainer/trainer.py
import logging
import torch
from pathlib import Path
from tqdm import tqdm
from losses.loss import Loss
from evaluators.evaluator import Evaluator
from utils.serialization import save_checkpoint
from utils.meters import AverageMeter

class Trainer:
    def __init__(self, model, args, monitor=None, runner=None):
        # 初始化训练器，设置模型、参数和设备
        self.model = model
        self.args = args
        self.monitor = monitor  # 添加监控器
        self.runner = runner  # 添加runner引用以便调用freeze方法
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 定义默认损失权重（与loss.py保持一致）
        default_loss_weights = {
            'info_nce': 1.0, 'cls': 0.05, 'cloth_semantic': 0.5, 
            'orthogonal': 0.1, 'gate_adaptive': 0.05
        }
        # 从配置文件获取损失权重，合并默认值
        loss_weights = getattr(args, 'disentangle', {}).get('loss_weights', default_loss_weights)
        # 确保所有必要的键都存在
        for key, value in default_loss_weights.items():
            if key not in loss_weights:
                loss_weights[key] = value
        self.combined_loss = Loss(temperature=0.1, weights=loss_weights).to(self.device)
        self.scaler = torch.amp.GradScaler('cuda', enabled=args.fp16) if self.device.type == 'cuda' else None
        if args.fp16 and self.device.type != 'cuda':
            logging.warning("FP16 is enabled but no CUDA device is available. Disabling mixed precision.")

    def run(self, inputs, epoch, batch_idx, total_batches):
        # 执行单次训练步骤，计算所有损失
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
            # 训练时不需要注意力图，return_attention=False（默认值）
            outputs = self.model(image=image, cloth_instruction=cloth_captions, id_instruction=id_captions)

            # 训练时模型返回 10 个输出（不包含注意力图）
            if len(outputs) != 10:
                raise ValueError(f"Expected 10 model outputs during training, got {len(outputs)}")

            image_feats, id_text_feats, fused_feats, id_logits, id_embeds, \
            cloth_embeds, cloth_text_embeds, cloth_image_embeds, gate, gate_weights = outputs

            loss_dict = self.combined_loss(
                image_embeds=image_feats, id_text_embeds=id_text_feats, fused_embeds=fused_feats,
                id_logits=id_logits, id_embeds=id_embeds, cloth_embeds=cloth_embeds,
                cloth_text_embeds=cloth_text_embeds, cloth_image_embeds=cloth_image_embeds,
                pids=pid, is_matched=is_matched, epoch=epoch, gate=gate
            )

        # 记录模型内部状态信息
        if self.monitor and batch_idx % 200 == 0:  # 每200个批次记录一次详细信息
            self.monitor.log_feature_statistics(image_feats, "image_features")
            self.monitor.log_feature_statistics(id_text_feats, "id_text_features")
            self.monitor.log_feature_statistics(fused_feats, "fused_features")
            self.monitor.log_feature_statistics(id_embeds, "identity_embeds")
            self.monitor.log_feature_statistics(cloth_embeds, "clothing_embeds")
            self.monitor.log_feature_statistics(cloth_text_embeds, "cloth_text_features")
            self.monitor.log_feature_statistics(cloth_image_embeds, "cloth_image_features")

            if gate is not None:
                self.monitor.log_gate_weights(gate, "disentangle_gate")
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
                image_feats, id_text_feats, _, _, _, _, _, _, _, gate_weights = outputs
                sim = torch.matmul(image_feats, id_text_feats.t())
                pos_sim = sim.diag().mean().item()
                neg_sim = sim[~torch.eye(sim.shape[0], dtype=bool, device=self.device)].mean().item()
                scale = self.model.scale
                return pos_sim, neg_sim, None, scale
        self.model.train()
        return None, None, None, None

    def _format_loss_display(self, loss_meters):
        # 格式化损失显示，按指定顺序排列并隐藏特定项
        display_order = ['info_nce', 'cls', 'cloth_semantic', 'orthogonal', 'gate_adaptive', 'total']
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

        for epoch in range(1, self.args.epochs + 1):
            # 【渐进解冻策略】在特定epoch检查并调整冻结状态和优化器
            stage_changed = False
            if self.runner:
                if epoch == 6:  # Stage 2: 解冻BERT和ViT后4层
                    print("\n=== Progressive Unfreezing: Stage 2 ===")
                    print("Epoch 6-20: Unfreezing BERT and ViT last 4 layers (layer 8-11)")
                    self.runner.freeze_bert_layers(self.model, unfreeze_from_layer=8)
                    self.runner.freeze_vit_layers(self.model, unfreeze_from_layer=8)
                    optimizer = self.runner.build_optimizer(self.model, stage=2)
                    lr_scheduler = self.runner.build_scheduler(optimizer)
                    stage_changed = True
                elif epoch == 21:  # Stage 3: 解冻BERT和ViT后8层
                    print("\n=== Progressive Unfreezing: Stage 3 ===")
                    print("Epoch 21-40: Unfreezing BERT and ViT last 8 layers (layer 4-11)")
                    self.runner.freeze_bert_layers(self.model, unfreeze_from_layer=4)
                    self.runner.freeze_vit_layers(self.model, unfreeze_from_layer=4)
                    optimizer = self.runner.build_optimizer(self.model, stage=3)
                    lr_scheduler = self.runner.build_scheduler(optimizer)
                    stage_changed = True
                elif epoch == 41:  # Stage 4: 全部解冻
                    print("\n=== Progressive Unfreezing: Stage 4 ===")
                    print("Epoch 41-60: Unfreezing all BERT and ViT layers")
                    self.runner.freeze_bert_layers(self.model, unfreeze_from_layer=0)
                    self.runner.freeze_vit_layers(self.model, unfreeze_from_layer=0)
                    optimizer = self.runner.build_optimizer(self.model, stage=4)
                    lr_scheduler = self.runner.build_scheduler(optimizer)
                    stage_changed = True
                elif epoch == 61:  # Stage 5: 降低学习率
                    print("\n=== Progressive Unfreezing: Stage 5 ===")
                    print("Epoch 61-80: Fine-tuning with reduced learning rate")
                    optimizer = self.runner.build_optimizer(self.model, stage=5)
                    lr_scheduler = self.runner.build_scheduler(optimizer)
                    stage_changed = True
                    
            if stage_changed and self.monitor:
                self.monitor.logger.info(f"Stage changed at epoch {epoch}")
            
            # 打印上一个 epoch 的平均损失
            if epoch > 1:
                avg_losses = self._format_loss_display(loss_meters)
                if avg_losses:
                    # 将平均损失记录到日志，而不是打印到终端
                    if self.monitor:
                        self.monitor.logger.info(f"[Epoch {epoch-1} Avg Loss:] : {', '.join(avg_losses)}")
                    else:
                        print(f"[Avg Loss:] : {', '.join(avg_losses)}")

            # 重置损失记录器
            for meter in loss_meters.values():
                meter.reset()

            progress_bar = tqdm(
                train_loader, desc=f"[Epoch {epoch}/{self.args.epochs}] Training",
                dynamic_ncols=True, leave=True, total=total_batches
            )

            for i, inputs in enumerate(progress_bar):
                optimizer.zero_grad()
                loss_dict = self.run(inputs, epoch, i, total_batches)
                loss = loss_dict['total']

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    
                    # 梯度裁剪：放宽限制，允许分类损失更快下降
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                    # 记录梯度流动（每1000个batch记录一次）
                    if self.monitor and i % 1000 == 0:
                        self.monitor.log_gradient_flow(self.model)
                    # 记录基础梯度信息（每100个batch）
                    elif self.monitor and i % 100 == 0:
                        self.monitor.log_gradients(self.model, f"epoch_{epoch}_batch_{i}")

                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    
                    # 梯度裁剪：放宽限制
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

                    # 记录梯度流动（每1000个batch记录一次）
                    if self.monitor and i % 1000 == 0:
                        self.monitor.log_gradient_flow(self.model)
                    # 记录基础梯度信息（每100个batch）
                    elif self.monitor and i % 100 == 0:
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
                if self.monitor and i % 50 == 0:  # 每50个批次记录一次
                    current_lr = optimizer.param_groups[0]['lr']
                    self.monitor.log_batch_info(epoch, i, total_batches,
                                              {k: v.avg for k, v in loss_meters.items()},
                                              current_lr)

            progress_bar.close()
            
            # 只在stage未改变时调用lr_scheduler.step()
            if not stage_changed:
                lr_scheduler.step()

            # 记录epoch信息
            if self.monitor:
                epoch_metrics = {k: v.avg for k, v in loss_meters.items()}
                self.monitor.log_epoch_info(epoch, self.args.epochs, epoch_metrics)

            # 每个epoch结束后进行评估
            if query_loader and gallery_loader:
                # 评估模型
                evaluator = Evaluator(self.model, args=self.args)
                metrics = evaluator.evaluate(
                    query_loader, gallery_loader, query_loader.dataset.data,
                    gallery_loader.dataset.data, checkpoint_path=None, epoch=epoch
                )

                current_mAP = metrics['mAP']

                # 记录评估结果到日志
                if self.monitor:
                    self.monitor.logger.info(f"Epoch {epoch} - Evaluation Results: {metrics}")
                
                # 按照用户要求，在终端竖列显示评估指标
                print(f"\n[Epoch {epoch} Evaluation Results]")
                print(f"  mAP:    {metrics['mAP']:.4f}")
                print(f"  Rank-1: {metrics['rank1']:.4f}")
                print(f"  Rank-5: {metrics['rank5']:.4f}")
                print(f"  Rank-10: {metrics['rank10']:.4f}\n")

                # 保存最优检查点
                if current_mAP > best_mAP:
                    best_mAP = current_mAP

                    # 生成最佳检查点路径
                    if checkpoint_dir:
                        # 根据数据集名称确定模型文件名
                        dataset_short_name = self._get_dataset_name()

                        # 获取数据集的标准名称用于目录结构
                        if hasattr(self.args, 'dataset_configs') and self.args.dataset_configs:
                            dataset_full_name = self.args.dataset_configs[0]['name'].lower()
                            if 'cuhk' in dataset_full_name:
                                dataset_dir_name = 'cuhk_pedes'
                            elif 'rstp' in dataset_full_name:
                                dataset_dir_name = 'rstp'
                            elif 'icfg' in dataset_full_name:
                                dataset_dir_name = 'icfg'
                            else:
                                dataset_dir_name = dataset_full_name
                        else:
                            dataset_dir_name = dataset_short_name

                        # 构建检查点保存路径为 PROJECT_ROOT/log/DATASET_DIR_NAME/model/best_DATASET_SHORTNAME.pth
                        # 获取项目根目录（通过当前脚本路径向上两级）
                        script_dir = Path(__file__).parent  # trainers/
                        project_root = script_dir.parent   # 项目根目录

                        # 根据数据集名称确定正确的目录名，不依赖于传入的checkpoint_dir
                        if hasattr(self.args, 'dataset_configs') and self.args.dataset_configs:
                            dataset_full_name = self.args.dataset_configs[0]['name'].lower()
                            if 'cuhk' in dataset_full_name:
                                dataset_dir_name_correct = 'cuhk'
                            elif 'rstp' in dataset_full_name:
                                dataset_dir_name_correct = 'rstp'
                            elif 'icfg' in dataset_full_name:
                                dataset_dir_name_correct = 'icfg'
                            else:
                                dataset_dir_name_correct = dataset_short_name
                        else:
                            dataset_dir_name_correct = dataset_short_name

                        log_base_path = project_root / 'log'

                        model_dir = log_base_path / dataset_dir_name_correct / 'model'
                        model_dir.mkdir(parents=True, exist_ok=True)

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
                            self.monitor.logger.info(f"New best checkpoint saved: {best_checkpoint_path}, mAP: {best_mAP:.4f}")
                    else:
                        if self.monitor:
                            self.monitor.logger.warning("checkpoint_dir not provided, cannot save best checkpoint")

        # 记录最终结果
        if self.monitor:
            self.monitor.logger.info(f"Training completed. Best mAP: {best_mAP:.4f}")

        # 打印最终平均损失
        avg_losses = self._format_loss_display(loss_meters)
        if avg_losses:
            # 将最终平均损失记录到日志，而不是打印到终端
            if self.monitor:
                self.monitor.logger.info(f"[Final Avg Loss:] : {', '.join(avg_losses)}")
            else:
                print(f"[Avg Loss:] : {', '.join(avg_losses)}")

        if best_checkpoint_path:
            logging.info(f"Final best checkpoint: {best_checkpoint_path}, mAP: {best_mAP:.4f}")

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