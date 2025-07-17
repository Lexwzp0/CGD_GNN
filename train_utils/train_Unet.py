# python3 /home/wangzepeng/CGD_GNN/train_utils/train_Unet.py
import sys
import os
import argparse
from torch.utils.data import random_split

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 src 目录在当前目录的上一级
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
# from data.pyg_dataToGraph import DataToGraph
from src.utils.pyg_dataToGraph import DataToGraph # 更新路径
import torch.nn.functional as F
from matplotlib import pyplot as plt
# from src.Unet import ConditionalDiffusionUNet
from src.models.Unet import ConditionalDiffusionUNet # 更新路径
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# from src.gaussian_diffusion import GuidedGaussianDiffusion
from src.models.gaussian_diffusion import GuidedGaussianDiffusion # 更新路径

def main(args):
    # -- 1. 设置和配置 --
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # -- 2. 数据加载和准备 --
    dataset = DataToGraph(
        raw_data_path=args.data_path,
        dataset_name='TFF' + '.mat'
    )
    num_classes = dataset.num_classes

    x0, labels = [], []
    for data in dataset:
        x0.append(data.x)
        labels.append(data.y)

    x0 = torch.stack(x0).unsqueeze(1)
    labels = torch.stack(labels)

    print(f"原始数据集大小: {len(dataset)}")
    print(f"类别数: {num_classes}")
    print("X0 shape:", x0.shape)
    print("Labels shape:", labels.shape)

    # 划分训练集和验证集
    full_dataset = TensorDataset(x0, labels)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # -- 3. 模型、优化器和调度器初始化 --
    model = ConditionalDiffusionUNet(
        num_classes=num_classes,
        time_dim=128,
        label_dim=64
    ).to(device)
    
    diffusion = GuidedGaussianDiffusion(
        num_steps=args.diffusion_steps
    )
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # -- 4. 训练循环 --
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [训练]", unit="batch")
        
        for x_batch, label_batch in pbar:
            x_batch, label_batch = x_batch.to(device), label_batch.to(device)
            
            t = torch.randint(0, diffusion.num_steps, (x_batch.size(0),), device=device).long()
            
            losses = diffusion.training_losses(model=model, x_start=x_batch, t=t, batch_labels=label_batch)
            loss = losses["loss"].mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({
                'Loss': loss.item(),
                'MSE': losses['mse'].mean().item(),
                'VB': losses['vb'].mean().item()
            })

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, label_batch in val_loader:
                x_batch, label_batch = x_batch.to(device), label_batch.to(device)
                t = torch.randint(0, diffusion.num_steps, (x_batch.size(0),), device=device).long()
                losses = diffusion.training_losses(model=model, x_start=x_batch, t=t, batch_labels=label_batch)
                loss = losses["loss"].mean()
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        scheduler.step()

        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4e} | Val Loss: {avg_val_loss:.4e} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_filename = 'best_unet.pth'
            save_path = os.path.join(args.model_dir, model_filename)
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最佳U-Net模型到 {save_path} | 验证损失: {best_val_loss:.4f}")

    # -- 5. 结果可视化和保存 --
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("U-Net 训练和验证损失曲线")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(args.plot_dir, 'unet_loss_curve.png')
    plt.savefig(plot_path)
    plt.show()

    print("🎉 训练完成!")
    print(f"📈 训练曲线图已保存到 {plot_path}")
    print(f"🏆 最佳验证损失: {best_val_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练用于条件扩散的U-Net模型")
    
    # 路径和目录参数
    parser.add_argument('--data_path', type=str, default='/data/wangzepeng/raw', help='原始数据的路径')
    parser.add_argument('--model_dir', type=str, default='/data/wangzepeng/models/unet', help='U-Net模型检查点的保存目录')
    parser.add_argument('--plot_dir', type=str, default='/home/wangzepeng/CGD_GNN/results/unet', help='训练曲线图的保存目录')

    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=10000, help='训练的总轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='训练的批量大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪的阈值')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集所占的比例')
    
    # 模型相关参数
    parser.add_argument('--diffusion_steps', type=int, default=64, help='扩散过程的步数')

    args = parser.parse_args()
    main(args)