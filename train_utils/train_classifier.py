# python3 /home/wangzepeng/CGD_GNN/train_utils/train_classifier.py
import sys
import os
# 新增模块
import argparse
from torch.utils.data import random_split

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 src 目录在当前目录的上一级
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# 现在尝试导入
# from src.gaussian_diffusion import GuidedGaussianDiffusion
# from src.classifier import UNetClassifier
# from data.pyg_dataToGraph import DataToGraph
from src.models.gaussian_diffusion import GuidedGaussianDiffusion
from src.models.classifier import UNetClassifier
from src.utils.pyg_dataToGraph import DataToGraph
import torch.nn.functional as F
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

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
    
    torch.manual_seed(42) # 保证划分的可复现性
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # -- 3. 模型、优化器和调度器初始化 --
    classifier = UNetClassifier(num_classes).to(device)
    diffusion = GuidedGaussianDiffusion(
        num_steps=args.diffusion_steps
    )
    optimizer = AdamW(classifier.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # -- 4. 训练循环 --
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    best_val_acc = 0.0

    for epoch in range(args.num_epochs):
        # --- 训练阶段 ---
        classifier.train()
        train_loss, train_correct, train_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.num_epochs} [训练]", unit="batch")
        
        for x_batch, label_batch in pbar:
            x_batch, label_batch = x_batch.to(device), label_batch.to(device)
            # 确保标签维度正确, [B, 1] -> [B]
            if label_batch.dim() > 1:
                label_batch = label_batch.squeeze(1)

            t = torch.randint(0, diffusion.num_steps, (x_batch.size(0),), device=device).long()
            noisy_batch = diffusion.q_sample(x_start=x_batch, t=t)
            logits = classifier(noisy_batch, t)
            
            loss = F.cross_entropy(logits, label_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
            _, predicted = torch.max(logits, 1)
            train_correct += (predicted == label_batch).sum().item()
            train_total += label_batch.size(0)
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{train_correct / train_total:.2%}"
            })

        avg_train_loss = train_loss / train_total
        avg_train_acc = train_correct / train_total
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)

        # --- 验证阶段 ---
        classifier.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x_batch, label_batch in val_loader:
                x_batch, label_batch = x_batch.to(device), label_batch.to(device)
                if label_batch.dim() > 1:
                    label_batch = label_batch.squeeze(1)

                t = torch.randint(0, diffusion.num_steps, (x_batch.size(0),), device=device).long()
                noisy_batch = diffusion.q_sample(x_start=x_batch, t=t)
                logits = classifier(noisy_batch, t)
                
                loss = F.cross_entropy(logits, label_batch)
                
                val_loss += loss.item() * x_batch.size(0)
                _, predicted = torch.max(logits, 1)
                val_correct += (predicted == label_batch).sum().item()
                val_total += label_batch.size(0)
        
        avg_val_loss = val_loss / val_total
        avg_val_acc = val_correct / val_total
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(avg_val_acc)
        
        scheduler.step()

        print(f"Epoch {epoch + 1:03d} | Train Loss: {avg_train_loss:.4e}, Acc: {avg_train_acc:.2%} | "
              f"Val Loss: {avg_val_loss:.4e}, Acc: {avg_val_acc:.2%} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e}")

        # 保存最佳模型
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            model_filename = 'best_noisy_classifier.pth'
            save_path = os.path.join(args.model_dir, model_filename)
            torch.save(classifier.state_dict(), save_path)
            print(f"✅ 保存最佳模型到 {save_path} | 验证准确率: {best_val_acc:.2%}")

    # -- 5. 结果可视化和保存 --
    plt.figure(figsize=(14, 6))
    plt.suptitle(f"Training Curves", fontsize=16)

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
    
    plot_filename = 'training_curves.png'
    plot_path = os.path.join(args.plot_dir, plot_filename)
    plt.savefig(plot_path)
    plt.show()
    print(f"🎉 训练完成! 训练曲线图已保存到 {plot_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="为引导扩散模型训练分类器")
    
    # 路径和目录参数
    parser.add_argument('--data_path', type=str, default='/data/wangzepeng/raw', help='原始数据的路径')
    parser.add_argument('--model_dir', type=str, default='/data/wangzepeng/models/classifier', help='模型检查点的保存目录')
    parser.add_argument('--plot_dir', type=str, default='/home/wangzepeng/CGD_GNN/results/classifier', help='训练曲线图的保存目录')

    # 训练超参数
    parser.add_argument('--num_epochs', type=int, default=1000, help='训练的总轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='训练的批量大小')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪的阈值')
    parser.add_argument('--val_split', type=float, default=0.2, help='验证集所占的比例')
    
    # 模型相关参数
    parser.add_argument('--diffusion_steps', type=int, default=64, help='扩散过程的步数')

    args = parser.parse_args()
    main(args)