# python3 /home/wangzepeng/CGD_GNN/train_utils/train_Unet_sample.py
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
from src.models.classifier import UNetClassifier # 重新引入
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
# from src.gaussian_diffusion import GuidedGaussianDiffusion
from src.models.gaussian_diffusion import GuidedGaussianDiffusion # 更新路径
import seaborn as sns # 新增，用于绘图
import numpy as np # 新增，用于数学运算

def main(args):
    # -- 1. 设置和配置 --
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    # 移除beta调度相关的目录区分
    heatmap_dir = os.path.join(args.plot_dir, 'epoch_heatmaps')
    os.makedirs(heatmap_dir, exist_ok=True)
    
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
    
    # 重新加载评估分类器作为占位符
    print(f"加载评估分类器: {args.eval_classifier_path}")
    eval_classifier = UNetClassifier(num_classes).to(device)
    try:
        eval_classifier.load_state_dict(torch.load(args.eval_classifier_path, map_location=device))
        eval_classifier.eval()
    except Exception as e:
        print(f"加载评估分类器失败: {e}"); sys.exit(1)
        
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

        # --- Intermittent Sampling and Visualization ---
        if (epoch + 1) % args.sample_interval == 0 or (epoch + 1) == args.num_epochs:
            run_intermittent_eval(model, diffusion, eval_classifier, device, num_classes, args, epoch, heatmap_dir)
            
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 移除文件名中的beta调度信息
            model_filename = 'best_unet.pth'
            save_path = os.path.join(args.model_dir, model_filename)
            torch.save(model.state_dict(), save_path)
            print(f"✅ 保存最佳U-Net模型到 {save_path} | 验证损失: {best_val_loss:.4e}")

    # -- 5. 结果可视化和保存 --
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("U-Net Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 移除文件名中的beta调度信息
    plot_filename = 'unet_loss_curve.png'
    plot_path = os.path.join(args.plot_dir, plot_filename)
    plt.savefig(plot_path)
    plt.show()

    print("🎉 训练完成!")
    print(f"📈 训练曲线图已保存到 {plot_path}")
    print(f"🏆 最佳验证损失: {best_val_loss:.4e}")

def run_intermittent_eval(unet_model, diffusion, eval_classifier, device, num_classes, args, epoch, heatmap_dir):
    print(f"\n--- Running intermittent evaluation at epoch {epoch + 1} ---")
    unet_model.eval()
    
    # 使用无引导的采样来评估U-Net本身的能力
    original_scale = diffusion.classifier_scale
    diffusion.classifier_scale = 0.0
    
    # 设置子图布局
    num_cols = min(num_classes, 4) # 每行最多4个
    num_rows = (num_classes + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    axes = axes.flatten() # 将2D数组展平为1D，方便索引

    try:
        with torch.no_grad():
            for class_idx in range(num_classes):
                # 为每个类别生成一个样本进行可视化
                target_labels = torch.full((1,), class_idx, device=device).long()
                
                generated = diffusion.p_sample_loop(
                    model=unet_model,
                    classifier=eval_classifier, # 传入分类器作为占位符
                    shape=(1, 1, 24, 50),
                    batch_labels=target_labels,
                    progress=False
                )
                
                # 获取样本张量并移至CPU
                sample_tensor = generated[0, 0].cpu().numpy()
                ax = axes[class_idx]

                # 在对应的子图上绘制热力图
                sns.heatmap(sample_tensor, cmap="viridis", ax=ax)
                ax.set_title(f'Class {class_idx}')
                ax.set_xlabel("Features")
                ax.set_ylabel("Nodes")
        
        # 隐藏多余的子图
        for i in range(num_classes, len(axes)):
            axes[i].set_visible(False)
            
        fig.suptitle(f'Epoch {epoch + 1} - Generated Samples', fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.96]) # 调整布局以适应主标题
        
        # 保存整个图像
        heatmap_path = os.path.join(heatmap_dir, f"samples_epoch_{epoch+1}.png")
        plt.savefig(heatmap_path)
        plt.close(fig)

    finally:
        # 恢复原始的分类器引导强度
        diffusion.classifier_scale = original_scale

    print(f"Saved intermittent sample heatmaps for epoch {epoch + 1} to: {heatmap_dir}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="训练用于条件扩散的U-Net模型，并定期进行采样评估")
    
    # 路径和目录参数
    parser.add_argument('--data_path', type=str, default='/data/wangzepeng/raw', help='原始数据的路径')
    parser.add_argument('--model_dir', type=str, default='/data/wangzepeng/models/unet_sample', help='U-Net模型检查点的保存目录')
    parser.add_argument('--plot_dir', type=str, default='/home/wangzepeng/CGD_GNN/results/unet_sample', help='训练曲线图的保存目录')
    parser.add_argument('--eval_classifier_path', type=str, default='/data/wangzepeng/models/classifier/best_eval_classifier.pth',
                        help='用于评估/占位的(clean)分类器模型路径')

    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练的总轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='训练的批量大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪的阈值')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集所占的比例')
    
    # 采样评估参数
    parser.add_argument('--sample_interval', type=int, default=10, help='每隔多少个epoch进行一次采样评估')
    
    # 模型相关参数
    parser.add_argument('--diffusion_steps', type=int, default=64, help='扩散过程的步数')

    args = parser.parse_args()
    main(args)