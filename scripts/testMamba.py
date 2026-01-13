#!/usr/bin/env python3
"""
Mamba SSM 模型测试脚本
功能:验证Mamba模型的基本功能,包括前向传播和反向传播
"""

import torch
import torch.nn as nn
from typing import Optional
import time

try:
    from mamba_ssm import Mamba
except ImportError:
    print("错误：未安装 mamba-ssm 库")
    print("请运行：pip install mamba-ssm causal-conv1d")
    exit(1)


def setup_device() -> torch.device:
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ 使用CUDA设备: {torch.cuda.get_device_name()}")
        print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        device = torch.device("cpu")
        print("⚠ 使用CPU设备 (建议使用GPU加速)")
    return device


def create_mamba_model(d_model: int = 128, d_state: int = 16, 
                      d_conv: int = 4, expand: int = 2, 
                      device: Optional[torch.device] = None) -> nn.Module:
    """创建Mamba模型"""
    if device is None:
        device = setup_device()
    
    try:
        model = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        ).to(device)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Mamba模型创建成功")
        print(f"  模型参数: {total_params:,} (可训练: {trainable_params:,})")
        print(f"  特征维度: {d_model}, 状态维度: {d_state}")
        
        return model
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        raise


def test_forward_pass(model: nn.Module, batch_size: int = 4, 
                     seq_length: int = 64, d_model: int = 128,
                     device: torch.device = None) -> torch.Tensor:
    """测试前向传播"""
    print(f"\n--- 前向传播测试 ---")
    
    # 创建输入数据
    input_tensor = torch.randn(batch_size, seq_length, d_model, device=device)
    print(f"输入形状: {input_tensor.shape}")
    
    # 执行前向传播并计时
    model.eval()
    start_time = time.time()
    
    with torch.no_grad():
        output = model(input_tensor)
    
    forward_time = time.time() - start_time
    
    # 验证输出
    expected_shape = (batch_size, seq_length, d_model)
    if output.shape == expected_shape:
        print(f"✓ 输出形状正确: {output.shape}")
    else:
        print(f"❌ 输出形状错误: 期望 {expected_shape}, 实际 {output.shape}")
    
    print(f"✓ 前向传播耗时: {forward_time*1000:.2f}ms")
    print(f"  输出样本 (前5个值): {output[0, 0, :5].cpu().numpy()}")
    print(f"  输出统计 - 均值: {output.mean():.4f}, 标准差: {output.std():.4f}")
    
    return output


def test_backward_pass(model: nn.Module, batch_size: int = 4, 
                      seq_length: int = 64, d_model: int = 128,
                      device: torch.device = None) -> None:
    """测试反向传播"""
    print(f"\n--- 反向传播测试 ---")
    
    # 创建需要梯度的输入
    input_tensor = torch.randn(batch_size, seq_length, d_model, 
                              device=device, requires_grad=True)
    
    model.train()
    start_time = time.time()
    
    # 前向传播
    output = model(input_tensor)
    
    # 计算损失并反向传播
    loss = output.mean()  # 使用均值而不是求和，避免梯度爆炸
    loss.backward()
    
    backward_time = time.time() - start_time
    
    # 检查梯度
    if input_tensor.grad is not None:
        grad_norm = input_tensor.grad.norm().item()
        print(f"✓ 反向传播成功")
        print(f"  梯度范数: {grad_norm:.6f}")
        print(f"  损失值: {loss.item():.6f}")
    else:
        print("❌ 梯度计算失败")
    
    # 检查模型参数梯度
    param_grads = [p.grad.norm().item() for p in model.parameters() 
                   if p.grad is not None]
    if param_grads:
        print(f"  模型参数梯度范数: 最大={max(param_grads):.6f}, 最小={min(param_grads):.6f}")
    
    print(f"✓ 反向传播耗时: {backward_time*1000:.2f}ms")


def benchmark_model(model: nn.Module, device: torch.device, 
                   batch_size: int = 4, seq_length: int = 64, 
                   d_model: int = 128, num_runs: int = 10) -> None:
    """性能基准测试"""
    print(f"\n--- 性能基准测试 ({num_runs}次运行) ---")
    
    model.eval()
    input_tensor = torch.randn(batch_size, seq_length, d_model, device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_tensor)
    
    # 实际测试
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    total_time = time.time() - start_time
    
    avg_time = total_time / num_runs
    throughput = batch_size / avg_time
    
    print(f"✓ 平均推理时间: {avg_time*1000:.2f}ms")
    print(f"✓ 吞吐量: {throughput:.1f} samples/sec")


def main():
    """主函数"""
    print("🔍 Mamba SSM 模型测试")
    print("=" * 50)
    
    # 配置参数
    config = {
        'd_model': 128,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'batch_size': 4,
        'seq_length': 64
    }
    
    try:
        # 1. 设置设备
        device = setup_device()
        
        # 2. 创建模型
        model = create_mamba_model(
            d_model=config['d_model'],
            d_state=config['d_state'],
            d_conv=config['d_conv'],
            expand=config['expand'],
            device=device
        )
        
        # 3. 测试前向传播
        _ = test_forward_pass(
            model, config['batch_size'], 
            config['seq_length'], config['d_model'], device
        )
        
        # 4. 测试反向传播
        test_backward_pass(
            model, config['batch_size'], 
            config['seq_length'], config['d_model'], device
        )
        
        # 5. 性能基准测试
        benchmark_model(model, device, 
                       config['batch_size'], config['seq_length'], 
                       config['d_model'])
        
        print(f"\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        raise


if __name__ == "__main__":
    main()