# python3 /home/wangzepeng/CGD_GNN/aug_utils/simply_mix.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 添加项目根目录
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 假设 GCN 模型和 DataToGraph 工具类可以被正确导入
from src.models.GCN_model import GCN
from src.utils.pyg_dataToGraph import DataToGraph

# --- 1. 核心功能函数 ---

def load_datasets(raw_data_path, generated_data_path):
    """加载原始数据集和生成的数据集"""
    print("--- 1. 加载数据 ---")
    # 加载原始数据
    original_dataset = DataToGraph(raw_data_path=raw_data_path, dataset_name='TFF.mat')
    print(f"原始数据集加载完成，共 {len(original_dataset)} 个样本。")

    # 加载生成数据
    generated_dataset = []
    if os.path.exists(generated_data_path):
        try:
            # 使用 weights_only=False 以加载自定义的PyG图对象
            loaded_data = torch.load(generated_data_path, weights_only=False)
            print(f"成功加载文件: {generated_data_path}")
            print(f"  - 文件中包含的数据类型: {type(loaded_data)}")

            # 检查加载的数据是字典还是列表
            if isinstance(loaded_data, dict):
                print("  - 检测到数据为字典格式。正在提取 'samples' 和 'target_labels'...")
                gen_x = loaded_data.get('samples')
                gen_labels = loaded_data.get('target_labels')
                
                if gen_x is not None and gen_labels is not None:
                    template_graph = original_dataset[0]
                    for i in range(len(gen_x)):
                        new_graph = template_graph.clone()
                        new_graph.x = gen_x[i].squeeze(0) if gen_x[i].dim() == 3 else gen_x[i]
                        new_graph.y = torch.tensor([gen_labels[i]], dtype=torch.long)
                        generated_dataset.append(new_graph)
                    print(f"  - 已从字典成功转换 {len(generated_dataset)} 个图样本。")
                else:
                    print("  - 警告: 字典中未找到 'samples' 或 'target_labels'。")

            elif isinstance(loaded_data, list):
                print("  - 检测到数据为列表格式。直接使用该列表。")
                # 确保列表不为空且包含的是图对象
                if loaded_data and hasattr(loaded_data[0], 'x') and hasattr(loaded_data[0], 'y'):
                    generated_dataset = loaded_data
                    print(f"  - 已成功加载 {len(generated_dataset)} 个图样本。")
                else:
                    print("  - 警告: 列表为空或内容不是预期的图对象格式。")
            else:
                print(f"  - 错误: 不支持的数据格式: {type(loaded_data)}")

        except Exception as e:
            print(f"错误: 加载或处理生成数据时发生意外: {e}")
    else:
        print(f"警告: 未找到生成数据文件: {generated_data_path}")
        
    return original_dataset, generated_dataset


def create_mixed_dataset(original_data, generated_data, mix_ratio):
    """
    数据增强策略：保留全部原始数据，并额外按比例增加生成数据。
    mix_ratio 控制使用多少比例的 *生成* 数据进行增强。
    """
    if mix_ratio <= 0 or not generated_data:
        print("混合比例为0或无生成数据，仅使用原始训练集。")
        return original_data

    # 确保 mix_ratio 不会超过1.0，如果超过则使用全部生成数据
    if mix_ratio > 1.0:
        print(f"警告: mix_ratio ({mix_ratio}) 大于1.0，将使用所有可用的生成数据。")
        mix_ratio = 1.0

    # 计算要从 *生成数据集* 中采样的数量
    num_generated_to_add = int(len(generated_data) * mix_ratio)
    
    # 从生成数据中随机采样
    additional_data = random.sample(generated_data, num_generated_to_add)
    
    # 将原始数据和额外数据合并
    mixed_dataset = original_data + additional_data
    
    random.shuffle(mixed_dataset)
    print(f"数据增强: 保留 {len(original_data)} 个原始样本，额外添加 {len(additional_data)} 个生成样本 (占总生成数据的 {mix_ratio:.0%})。")
    return mixed_dataset

def train_one_epoch(model, loader, optimizer, criterion, device):
    """执行一个完整的训练轮次"""
    model.train()
    total_loss = 0
    for batched_graph in loader:
        batched_graph = batched_graph.to(device)
        logits = model(batched_graph.x, batched_graph.edge_index, batched_graph.batch)
        loss = criterion(logits, batched_graph.y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def run_evaluation(model, loader, device):
    """在给定的数据集上进行评估，返回准确率和F1分数"""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batched_graph in loader:
            batched_graph = batched_graph.to(device)
            logits = model(batched_graph.x, batched_graph.edge_index, batched_graph.batch)
            y_true.append(batched_graph.y.cpu())
            y_pred.append(logits.argmax(dim=-1).cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return acc, f1_macro

def save_results(output_dir, model, metrics_df, args, current_ratio):
    """保存模型、指标CSV和结果图表"""
    print("\n--- 4. 保存结果 ---")
    # 为当前比例创建一个子目录
    ratio_dir = os.path.join(output_dir, f"ratio_{current_ratio:.2f}")
    os.makedirs(ratio_dir, exist_ok=True)
    
    # 构造文件名
    filename_prefix = f"mix_{current_ratio:.2f}"
    
    # 1. 保存最终模型
    model_path = os.path.join(ratio_dir, f"gcn_model_final_{filename_prefix}.pth")
    torch.save(model.state_dict(), model_path)
    print(f"最终模型已保存至: {model_path}")
    
    # 2. 保存指标
    csv_path = os.path.join(ratio_dir, f"training_metrics_{filename_prefix}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"训练指标已保存至: {csv_path}")
    
    # 3. 绘制并保存图表
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 绘制损失曲线
    ax1.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training Loss (Ratio: {current_ratio})')
    ax1.legend()
    
    # 绘制评估指标曲线
    ax2.plot(metrics_df['epoch'], metrics_df['val_acc'], label='Validation Accuracy (Clean)', marker='o', linestyle='-')
    ax2.plot(metrics_df['epoch'], metrics_df['test_acc'], label='Test Accuracy (Clean)', marker='x', linestyle='--')
    ax2.plot(metrics_df['epoch'], metrics_df['noisy_test_acc'], label=f'Test Accuracy (Noisy, L={args.noise_level})', marker='s', linestyle=':')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Model Accuracy (Ratio: {current_ratio})')
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(ratio_dir, f"metrics_plot_{filename_prefix}.png")
    plt.savefig(plot_path)
    print(f"结果图表已保存至: {plot_path}")
    plt.close()

def summarize_all_experiments(results, output_dir, args):
    """在所有实验结束后，生成总结报告和对比图"""
    if not results:
        print("没有实验结果可供总结。")
        return

    results_df = pd.DataFrame(results)
    
    # 打印总结表格
    print("\n" + "="*60)
    print(" " * 18 + "跨混合比例实验总结")
    print("="*60)
    # 为了打印效果更好，格式化浮点数列
    float_cols = ['val_acc', 'val_f1', 'test_acc', 'test_f1', 'noisy_test_acc', 'noisy_test_f1']
    for col in float_cols:
        results_df[col] = results_df[col].map('{:.4f}'.format)
    print(results_df.to_string(index=False))
    print("="*60)
    
    # 保存总结到CSV
    summary_csv_path = os.path.join(output_dir, 'experiment_summary.csv')
    results_df.to_csv(summary_csv_path, index=False)
    print(f"\n总结报告已保存至: {summary_csv_path}")

    # 重新读取数据用于绘图（避免格式化问题）
    plot_df = pd.read_csv(summary_csv_path)
    
    # 绘制总结对比图
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(plot_df['mix_ratio'], plot_df['test_acc'], label='测试集准确率 (Clean)', marker='o', linestyle='-')
    ax.plot(plot_df['mix_ratio'], plot_df['noisy_test_acc'], label=f'OOD测试集准确率 (Noisy, L={args.noise_level})', marker='x', linestyle='--')
    
    ax.set_xlabel('混合比例 (Mix Ratio)')
    ax.set_ylabel('准确率 (Accuracy)')
    ax.set_title('模型性能 vs. 数据增强混合比例')
    ax.legend()
    ax.grid(True, which='both', linestyle='--')
    
    summary_plot_path = os.path.join(output_dir, 'summary_plot_by_ratio.png')
    plt.savefig(summary_plot_path)
    print(f"总结图表已保存至: {summary_plot_path}")
    plt.close()


# --- 2. 主函数 ---

def run_single_experiment(args, mix_ratio, original_dataset, generated_dataset, device):
    """为单个混合比例执行一次完整的训练和评估"""
    # 2. 划分数据集并创建 DataLoader
    print("\n--- 2. 划分数据集 ---")
    split_idx = original_dataset.get_idx_split()
    original_train = [original_dataset[i] for i in split_idx["train"].tolist()]
    original_val = [original_dataset[i] for i in split_idx["valid"].tolist()]
    original_test = [original_dataset[i] for i in split_idx["test"].tolist()]
    
    # 创建混合训练集
    mixed_train_dataset = create_mixed_dataset(original_train, generated_dataset, mix_ratio)
    print(f"混合比例: {mix_ratio}, 训练集大小: {len(mixed_train_dataset)}")

    # 创建带噪声的OOD测试集来模拟分布外数据
    noise_level = args.noise_level
    noisy_test_dataset = []
    for graph in original_test:
        noisy_graph = graph.clone()
        # 添加乘性高斯噪声
        noise = torch.randn_like(graph.x) * noise_level
        noisy_graph.x = graph.x * (1 + noise)
        noisy_test_dataset.append(noisy_graph)
    print(f"已创建带高斯噪声 (level={noise_level}) 的OOD测试集。")
    
    train_loader = DataLoader(mixed_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(original_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(original_test, batch_size=args.batch_size, shuffle=False)
    noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=args.batch_size, shuffle=False)

    # 3. 初始化模型、优化器和损失函数
    input_dim = original_dataset[0].x.size(1)
    num_classes = original_dataset.num_classes
    model = GCN(in_feats=input_dim, hidden_channels=args.hidden_channels, out_feats=num_classes, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # 4. 训练循环
    print("\n--- 3. 开始训练 ---")
    metrics_history = []
    best_val_acc = 0
    best_epoch = 0

    pbar = tqdm(range(1, args.epochs + 1), desc=f"训练中 (Ratio: {mix_ratio})")
    for epoch in pbar:
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_f1 = run_evaluation(model, val_loader, device)
        test_acc, test_f1 = run_evaluation(model, test_loader, device)
        noisy_test_acc, noisy_test_f1 = run_evaluation(model, noisy_test_loader, device)
        
        pbar.set_postfix({
            "Loss": f"{train_loss:.4f}",
            "Val Acc": f"{val_acc:.4f}",
            "Test Acc": f"{test_acc:.4f}",
            "Noisy Test Acc": f"{noisy_test_acc:.4f}"
        })
        
        metrics_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'noisy_test_acc': noisy_test_acc,
            'noisy_test_f1': noisy_test_f1
        })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            # 保存最佳模型的状态
            best_model_path = os.path.join(args.output_dir, f"ratio_{mix_ratio:.2f}", f"gcn_model_best_mix_{mix_ratio:.2f}.pth")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)

    print(f"\n训练完成! (Ratio: {mix_ratio})")
    
    # 5. 保存结果
    metrics_df = pd.DataFrame(metrics_history)
    save_results(args.output_dir, model, metrics_df, args, mix_ratio)
    
    # 打印并返回最佳结果
    if not metrics_df.empty:
        best_metrics = metrics_df[metrics_df['epoch'] == best_epoch].iloc[0].to_dict()
        print("\n--- 最佳验证集结果 (Epoch {}) ---".format(best_epoch))
        print(f"  - 验证集: Acc={best_metrics['val_acc']:.4f}, F1-Macro={best_metrics['val_f1']:.4f}")
        print(f"  - 测试集 (Clean): Acc={best_metrics['test_acc']:.4f}, F1-Macro={best_metrics['test_f1']:.4f}")
        print(f"  - 测试集 (Noisy): Acc={best_metrics['noisy_test_acc']:.4f}, F1-Macro={best_metrics['noisy_test_f1']:.4f}")
        return best_metrics
    return None

def main(args):
    """主执行流程，负责调度所有实验"""
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    
    # 1. 一次性加载所有数据
    original_dataset, generated_dataset = load_datasets(args.raw_data_dir, args.generated_data_path)
   
    # 2. 循环执行每个混合比例的实验
    all_experiment_results = []
    for ratio in sorted(args.mix_ratios):
        print(f"\n{'='*25} 开始实验, 混合比例: {ratio} {'='*25}")
        best_metrics = run_single_experiment(args, ratio, original_dataset, generated_dataset, device)
        if best_metrics:
            best_metrics['mix_ratio'] = ratio
            all_experiment_results.append(best_metrics)
    
    # 3. 在所有实验结束后，生成总结报告
    summarize_all_experiments(all_experiment_results, args.output_dir, args)


# --- 3. 命令行参数解析 ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用混合数据增强训练GCN模型，并对比不同混合比例的效果")
    
    # 路径参数
    parser.add_argument('--raw_data_dir', type=str, default='/data/wangzepeng/raw', help='包含TFF.mat的原始数据目录')
    parser.add_argument('--generated_data_path', type=str, default='/data/wangzepeng/synthesis/generated_graphs.pt', help='生成的图数据文件路径 (.pt)')
    parser.add_argument('--output_dir', type=str, default='/home/wangzepeng/CGD_GNN/results/aug_experiment', help='保存所有实验结果的目录')

    # 模型和训练参数
    parser.add_argument('--device', type=int, default=0, help="CUDA设备ID")
    parser.add_argument('--epochs', type=int, default=200, help="训练轮数")
    parser.add_argument('--lr', type=float, default=0.005, help="学习率")
    parser.add_argument('--batch_size', type=int, default=128, help="批量大小")
    parser.add_argument('--hidden_channels', type=int, default=64, help="GCN隐藏层维度")
    
    # 数据增强参数
    parser.add_argument('--mix_ratios', nargs='+', type=float, default=[0.0, 0.2, 0.5, 0.8, 1.0], 
                        help="一系列用于实验的混合比例 (例如: 0.0 0.5 1.0)")
    parser.add_argument('--noise_level', type=float, default=0.8, help="为模拟OOD数据添加的乘性高斯噪声水平")

    args = parser.parse_args()
    main(args)