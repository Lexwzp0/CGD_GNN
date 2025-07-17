# python3 /home/wangzepeng/CGD_GNN/aug_utils/simply_mix_enhanced.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import argparse
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 假设 GCN 模型和 DataToGraph 工具类可以被正确导入
from src.models.GCN_model import GCN
from src.utils.pyg_dataToGraph import DataToGraph

# --- 1. 设置随机种子和日志 ---

def set_random_seed(seed=42):
    """设置所有随机种子以确保实验可重现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子为: {seed}")

def setup_logging(output_dir, experiment_name):
    """设置日志记录"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# --- 2. 增强的训练监督类 ---

class TrainingMonitor:
    """训练过程监控类"""
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = 0
        self.best_epoch = 0
        self.wait = 0
        self.early_stop = False
        self.train_history = []
        self.val_history = []
        self.test_history = []
        self.noisy_test_history = []
        self.loss_history = []
        
    def update(self, epoch, train_loss, val_acc, val_f1, test_acc, test_f1, noisy_test_acc, noisy_test_f1):
        """更新监控指标"""
        self.loss_history.append(train_loss)
        self.train_history.append({'epoch': epoch, 'loss': train_loss})
        self.val_history.append({'epoch': epoch, 'acc': val_acc, 'f1': val_f1})
        self.test_history.append({'epoch': epoch, 'acc': test_acc, 'f1': test_f1})
        self.noisy_test_history.append({'epoch': epoch, 'acc': noisy_test_acc, 'f1': noisy_test_f1})
        
        # 早停检查
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_epoch = epoch
            self.wait = 0
            return True  # 新的最佳模型
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.early_stop = True
                print(f"\n早停触发！在第 {epoch} 轮停止训练（最佳轮次: {self.best_epoch}）")
            return False

    def get_best_metrics(self):
        """获取最佳轮次的所有指标"""
        if self.best_epoch > 0:
            return {
                'epoch': self.best_epoch,
                'train_loss': self.loss_history[self.best_epoch-1],
                'val_acc': self.val_history[self.best_epoch-1]['acc'],
                'val_f1': self.val_history[self.best_epoch-1]['f1'],
                'test_acc': self.test_history[self.best_epoch-1]['acc'],
                'test_f1': self.test_history[self.best_epoch-1]['f1'],
                'noisy_test_acc': self.noisy_test_history[self.best_epoch-1]['acc'],
                'noisy_test_f1': self.noisy_test_history[self.best_epoch-1]['f1']
            }
        return None

# --- 3. 增强的核心功能函数 ---

def load_datasets(raw_data_path, generated_data_path, logger):
    """加载原始数据集和生成的数据集"""
    logger.info("=== 开始加载数据集 ===")
    
    # 加载原始数据
    start_time = time.time()
    original_dataset = DataToGraph(raw_data_path=raw_data_path, dataset_name='TFF.mat')
    load_time = time.time() - start_time
    logger.info(f"原始数据集加载完成: {len(original_dataset)} 个样本 (耗时: {load_time:.2f}s)")
    
    # 数据集基本信息
    logger.info(f"数据集信息:")
    logger.info(f"  - 节点特征维度: {original_dataset[0].x.size(1)}")
    logger.info(f"  - 类别数量: {original_dataset.num_classes}")
    if hasattr(original_dataset, 'get_idx_split'):
        split_idx = original_dataset.get_idx_split()
        logger.info(f"  - 训练集: {len(split_idx['train'])} 样本")
        logger.info(f"  - 验证集: {len(split_idx['valid'])} 样本")
        logger.info(f"  - 测试集: {len(split_idx['test'])} 样本")

    # 加载生成数据
    generated_dataset = []
    if os.path.exists(generated_data_path):
        try:
            start_time = time.time()
            loaded_data = torch.load(generated_data_path, weights_only=False)
            load_time = time.time() - start_time
            logger.info(f"生成数据加载完成 (耗时: {load_time:.2f}s)")
            logger.info(f"数据类型: {type(loaded_data)}")

            if isinstance(loaded_data, dict):
                gen_x = loaded_data.get('samples')
                gen_labels = loaded_data.get('target_labels')
                
                if gen_x is not None and gen_labels is not None:
                    template_graph = original_dataset[0]
                    for i in range(len(gen_x)):
                        new_graph = template_graph.clone()
                        new_graph.x = gen_x[i].squeeze(0) if gen_x[i].dim() == 3 else gen_x[i]
                        new_graph.y = torch.tensor([gen_labels[i]], dtype=torch.long)
                        generated_dataset.append(new_graph)
                    logger.info(f"成功转换 {len(generated_dataset)} 个生成样本")
                    
                    # 分析生成数据的类别分布
                    gen_labels_list = [int(label) for label in gen_labels]
                    unique_labels, counts = np.unique(gen_labels_list, return_counts=True)
                    logger.info("生成数据类别分布:")
                    for label, count in zip(unique_labels, counts):
                        logger.info(f"  - 类别 {label}: {count} 样本")

            elif isinstance(loaded_data, list):
                if loaded_data and hasattr(loaded_data[0], 'x') and hasattr(loaded_data[0], 'y'):
                    generated_dataset = loaded_data
                    logger.info(f"直接加载 {len(generated_dataset)} 个图样本")
                else:
                    logger.warning("列表为空或格式不正确")
            else:
                logger.error(f"不支持的数据格式: {type(loaded_data)}")

        except Exception as e:
            logger.error(f"加载生成数据时发生错误: {e}")
    else:
        logger.warning(f"未找到生成数据文件: {generated_data_path}")
        
    return original_dataset, generated_dataset

def create_mixed_dataset(original_data, generated_data, mix_ratio, logger):
    """增强的数据混合策略"""
    logger.info(f"创建混合数据集 (mix_ratio={mix_ratio})")
    
    if mix_ratio <= 0 or not generated_data:
        logger.info("仅使用原始训练集")
        return original_data

    if mix_ratio > 1.0:
        logger.warning(f"mix_ratio ({mix_ratio}) > 1.0，将使用所有生成数据")
        mix_ratio = 1.0

    num_generated_to_add = int(len(generated_data) * mix_ratio)
    additional_data = random.sample(generated_data, num_generated_to_add)
    mixed_dataset = original_data + additional_data
    random.shuffle(mixed_dataset)
    
    # 详细的数据分布分析
    original_labels = [graph.y.item() for graph in original_data]
    additional_labels = [graph.y.item() for graph in additional_data]
    
    logger.info("数据混合详情:")
    logger.info(f"  - 原始样本: {len(original_data)}")
    logger.info(f"  - 添加生成样本: {len(additional_data)} (占生成数据的 {mix_ratio:.0%})")
    logger.info(f"  - 混合后总样本: {len(mixed_dataset)}")
    
    # 类别分布统计
    from collections import Counter
    orig_dist = Counter(original_labels)
    add_dist = Counter(additional_labels)
    logger.info("原始数据类别分布: " + str(dict(orig_dist)))
    logger.info("添加数据类别分布: " + str(dict(add_dist)))
    
    return mixed_dataset

def enhanced_evaluation(model, loader, device, class_names=None):
    """增强的评估函数，包含更多指标"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batched_graph in loader:
            batched_graph = batched_graph.to(device)
            logits = model(batched_graph.x, batched_graph.edge_index, batched_graph.batch)
            loss = criterion(logits, batched_graph.y)
            total_loss += loss.item()
            
            y_true.append(batched_graph.y.cpu())
            y_pred.append(logits.argmax(dim=-1).cpu())
            y_prob.append(F.softmax(logits, dim=-1).cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()
    y_prob = torch.cat(y_prob, dim=0).numpy()
    
    # 计算各种指标
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    avg_loss = total_loss / len(loader)
    
    # 每个类别的F1分数
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'loss': avg_loss,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def create_comprehensive_plots(output_dir, monitor, eval_results, current_ratio, args):
    """创建更全面的可视化图表"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. 训练历史图表 (2x2布局)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle(f'训练监控总览 (Mix Ratio: {current_ratio})', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(monitor.loss_history) + 1)
    val_accs = [h['acc'] for h in monitor.val_history]
    val_f1s = [h['f1'] for h in monitor.val_history]
    test_accs = [h['acc'] for h in monitor.test_history]
    noisy_test_accs = [h['acc'] for h in monitor.noisy_test_history]
    
    # 损失曲线
    axes[0, 0].plot(epochs, monitor.loss_history, 'b-', linewidth=2, label='训练损失')
    axes[0, 0].axvline(x=monitor.best_epoch, color='r', linestyle='--', alpha=0.7, label=f'最佳轮次 ({monitor.best_epoch})')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].set_title('训练损失曲线')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 准确率对比
    axes[0, 1].plot(epochs, val_accs, 'g-', linewidth=2, marker='o', markersize=3, label='验证集')
    axes[0, 1].plot(epochs, test_accs, 'b-', linewidth=2, marker='s', markersize=3, label='测试集(Clean)')
    axes[0, 1].plot(epochs, noisy_test_accs, 'r-', linewidth=2, marker='^', markersize=3, label='测试集(Noisy)')
    axes[0, 1].axvline(x=monitor.best_epoch, color='orange', linestyle='--', alpha=0.7)
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].set_title('准确率对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1分数曲线
    val_f1s = [h['f1'] for h in monitor.val_history]
    test_f1s = [h['f1'] for h in monitor.test_history]
    noisy_test_f1s = [h['f1'] for h in monitor.noisy_test_history]
    
    axes[1, 0].plot(epochs, val_f1s, 'g-', linewidth=2, marker='o', markersize=3, label='验证集')
    axes[1, 0].plot(epochs, test_f1s, 'b-', linewidth=2, marker='s', markersize=3, label='测试集(Clean)')
    axes[1, 0].plot(epochs, noisy_test_f1s, 'r-', linewidth=2, marker='^', markersize=3, label='测试集(Noisy)')
    axes[1, 0].axvline(x=monitor.best_epoch, color='orange', linestyle='--', alpha=0.7)
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('F1-Macro分数')
    axes[1, 0].set_title('F1分数对比')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 性能提升分析
    best_metrics = monitor.get_best_metrics()
    improvement_data = {
        '指标': ['验证集准确率', '测试集准确率', '测试集准确率(Noisy)', 'F1-Macro(验证)', 'F1-Macro(测试)', 'F1-Macro(Noisy)'],
        '数值': [best_metrics['val_acc'], best_metrics['test_acc'], best_metrics['noisy_test_acc'],
                best_metrics['val_f1'], best_metrics['test_f1'], best_metrics['noisy_test_f1']]
    }
    
    colors = ['green', 'blue', 'red', 'lightgreen', 'lightblue', 'lightcoral']
    bars = axes[1, 1].bar(range(len(improvement_data['指标'])), improvement_data['数值'], color=colors, alpha=0.8)
    axes[1, 1].set_xlabel('评估指标')
    axes[1, 1].set_ylabel('分数')
    axes[1, 1].set_title(f'最佳性能总览 (第{monitor.best_epoch}轮)')
    axes[1, 1].set_xticks(range(len(improvement_data['指标'])))
    axes[1, 1].set_xticklabels(improvement_data['指标'], rotation=45, ha='right')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, improvement_data['数值']):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"comprehensive_training_analysis_mix_{current_ratio:.2f}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 混淆矩阵
    if eval_results:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'混淆矩阵分析 (Mix Ratio: {current_ratio})', fontsize=14, fontweight='bold')
        
        datasets = ['验证集', '测试集(Clean)', '测试集(Noisy)']
        for idx, (name, result) in enumerate(zip(datasets, eval_results)):
            if result is not None:
                cm = confusion_matrix(result['y_true'], result['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
                axes[idx].set_title(f'{name}\n准确率: {result["accuracy"]:.3f}')
                axes[idx].set_xlabel('预测标签')
                axes[idx].set_ylabel('真实标签')
        
        plt.tight_layout()
        cm_path = os.path.join(output_dir, f"confusion_matrices_mix_{current_ratio:.2f}.png")
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()

def save_enhanced_results(output_dir, model, monitor, eval_results, args, current_ratio, logger):
    """保存增强的实验结果"""
    logger.info("保存实验结果...")
    
    # 创建输出目录
    ratio_dir = os.path.join(output_dir, f"ratio_{current_ratio:.2f}")
    os.makedirs(ratio_dir, exist_ok=True)
    
    # 1. 保存模型
    model_path = os.path.join(ratio_dir, f"gcn_model_final_mix_{current_ratio:.2f}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"模型已保存: {model_path}")
    
    # 2. 保存详细的训练历史
    detailed_history = []
    for epoch in range(len(monitor.loss_history)):
        detailed_history.append({
            'epoch': epoch + 1,
            'train_loss': monitor.loss_history[epoch],
            'val_acc': monitor.val_history[epoch]['acc'],
            'val_f1': monitor.val_history[epoch]['f1'],
            'test_acc': monitor.test_history[epoch]['acc'],
            'test_f1': monitor.test_history[epoch]['f1'],
            'noisy_test_acc': monitor.noisy_test_history[epoch]['acc'],
            'noisy_test_f1': monitor.noisy_test_history[epoch]['f1']
        })
    
    history_df = pd.DataFrame(detailed_history)
    history_path = os.path.join(ratio_dir, f"detailed_training_history_mix_{current_ratio:.2f}.csv")
    history_df.to_csv(history_path, index=False)
    logger.info(f"训练历史已保存: {history_path}")
    
    # 3. 保存实验配置
    config = {
        'mix_ratio': current_ratio,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'hidden_channels': args.hidden_channels,
        'noise_level': args.noise_level,
        'best_epoch': monitor.best_epoch,
        'early_stopped': monitor.early_stop,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(ratio_dir, f"experiment_config_mix_{current_ratio:.2f}.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 4. 保存评估报告
    if eval_results:
        for idx, (name, result) in enumerate(zip(['validation', 'test_clean', 'test_noisy'], eval_results)):
            if result is not None:
                # 分类报告
                report = classification_report(result['y_true'], result['y_pred'], output_dict=True)
                report_path = os.path.join(ratio_dir, f"classification_report_{name}_mix_{current_ratio:.2f}.json")
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
    
    # 5. 创建可视化图表
    create_comprehensive_plots(ratio_dir, monitor, eval_results, current_ratio, args)
    
    return monitor.get_best_metrics()

def run_single_experiment(args, mix_ratio, original_dataset, generated_dataset, device, logger):
    """运行单个混合比例的实验"""
    logger.info(f"\n{'='*25} 开始实验: Mix Ratio = {mix_ratio} {'='*25}")
    
    # 数据集划分
    split_idx = original_dataset.get_idx_split()
    original_train = [original_dataset[i] for i in split_idx["train"].tolist()]
    original_val = [original_dataset[i] for i in split_idx["valid"].tolist()]
    original_test = [original_dataset[i] for i in split_idx["test"].tolist()]
    
    # 创建混合训练集
    mixed_train_dataset = create_mixed_dataset(original_train, generated_dataset, mix_ratio, logger)
    
    # 创建OOD测试集
    noisy_test_dataset = []
    for graph in original_test:
        noisy_graph = graph.clone()
        noise = torch.randn_like(graph.x) * args.noise_level
        noisy_graph.x = graph.x * (1 + noise)
        noisy_test_dataset.append(noisy_graph)
    logger.info(f"创建OOD测试集: {len(noisy_test_dataset)} 样本 (噪声水平: {args.noise_level})")
    
    # 创建数据加载器
    train_loader = DataLoader(mixed_train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(original_val, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(original_test, batch_size=args.batch_size, shuffle=False)
    noisy_test_loader = DataLoader(noisy_test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 初始化模型
    input_dim = original_dataset[0].x.size(1)
    num_classes = original_dataset.num_classes
    model = GCN(in_feats=input_dim, hidden_channels=args.hidden_channels, 
                out_feats=num_classes, num_layers=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    logger.info(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化训练监控
    monitor = TrainingMonitor(patience=args.patience, min_delta=0.001)
    
    # 训练循环
    logger.info("开始训练...")
    start_time = time.time()
    
    pbar = tqdm(range(1, args.epochs + 1), desc=f"训练中 (Ratio: {mix_ratio})")
    for epoch in pbar:
        # 训练
        model.train()
        total_loss = 0
        for batched_graph in train_loader:
            batched_graph = batched_graph.to(device)
            logits = model(batched_graph.x, batched_graph.edge_index, batched_graph.batch)
            loss = criterion(logits, batched_graph.y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_loss = total_loss / len(train_loader)
        
        # 评估
        val_result = enhanced_evaluation(model, val_loader, device)
        test_result = enhanced_evaluation(model, test_loader, device)
        noisy_test_result = enhanced_evaluation(model, noisy_test_loader, device)
        
        # 更新监控
        is_best = monitor.update(epoch, train_loss, 
                               val_result['accuracy'], val_result['f1_macro'],
                               test_result['accuracy'], test_result['f1_macro'],
                               noisy_test_result['accuracy'], noisy_test_result['f1_macro'])
        
        # 保存最佳模型
        if is_best:
            best_model_path = os.path.join(args.output_dir, f"ratio_{mix_ratio:.2f}", 
                                         f"gcn_model_best_mix_{mix_ratio:.2f}.pth")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
        
        # 更新进度条
        pbar.set_postfix({
            "Loss": f"{train_loss:.4f}",
            "Val Acc": f"{val_result['accuracy']:.4f}",
            "Test Acc": f"{test_result['accuracy']:.4f}",
            "Noisy Acc": f"{noisy_test_result['accuracy']:.4f}",
            "Best": f"E{monitor.best_epoch}"
        })
        
        # 早停检查
        if monitor.early_stop:
            break
    
    training_time = time.time() - start_time
    logger.info(f"训练完成! 总耗时: {training_time:.2f}s, 最佳轮次: {monitor.best_epoch}")
    
    # 最终评估
    # 加载最佳模型进行最终评估
    best_model_path = os.path.join(args.output_dir, f"ratio_{mix_ratio:.2f}", 
                                 f"gcn_model_best_mix_{mix_ratio:.2f}.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info("已加载最佳模型进行最终评估")
    
    final_val_result = enhanced_evaluation(model, val_loader, device)
    final_test_result = enhanced_evaluation(model, test_loader, device)
    final_noisy_test_result = enhanced_evaluation(model, noisy_test_loader, device)
    
    eval_results = [final_val_result, final_test_result, final_noisy_test_result]
    
    # 保存结果
    best_metrics = save_enhanced_results(args.output_dir, model, monitor, eval_results, args, mix_ratio, logger)
    
    # 打印最终结果
    logger.info("\n" + "="*60)
    logger.info(f"实验完成 - Mix Ratio: {mix_ratio}")
    logger.info("="*60)
    logger.info(f"最佳轮次: {monitor.best_epoch}")
    logger.info(f"验证集: Acc={final_val_result['accuracy']:.4f}, F1={final_val_result['f1_macro']:.4f}")
    logger.info(f"测试集(Clean): Acc={final_test_result['accuracy']:.4f}, F1={final_test_result['f1_macro']:.4f}")
    logger.info(f"测试集(Noisy): Acc={final_noisy_test_result['accuracy']:.4f}, F1={final_noisy_test_result['f1_macro']:.4f}")
    logger.info("="*60)
    
    return best_metrics

def create_final_comparison_plots(results_df, output_dir, args):
    """创建最终的对比分析图表"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. 综合性能对比图
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('跨混合比例实验综合分析', fontsize=16, fontweight='bold')
    
    mix_ratios = results_df['mix_ratio']
    
    # 准确率对比
    axes[0, 0].plot(mix_ratios, results_df['val_acc'], 'g-o', linewidth=2, markersize=4, label='验证集')
    axes[0, 0].plot(mix_ratios, results_df['test_acc'], 'b-s', linewidth=2, markersize=4, label='测试集(Clean)')
    axes[0, 0].plot(mix_ratios, results_df['noisy_test_acc'], 'r-^', linewidth=2, markersize=4, label='测试集(Noisy)')
    axes[0, 0].set_xlabel('混合比例', fontsize=12)
    axes[0, 0].set_ylabel('准确率', fontsize=12)
    axes[0, 0].set_title('准确率 vs 混合比例', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1分数对比
    axes[0, 1].plot(mix_ratios, results_df['val_f1'], 'g-o', linewidth=2, markersize=4, label='验证集')
    axes[0, 1].plot(mix_ratios, results_df['test_f1'], 'b-s', linewidth=2, markersize=4, label='测试集(Clean)')
    axes[0, 1].plot(mix_ratios, results_df['noisy_test_f1'], 'r-^', linewidth=2, markersize=4, label='测试集(Noisy)')
    axes[0, 1].set_xlabel('混合比例', fontsize=12)
    axes[0, 1].set_ylabel('F1-Macro分数', fontsize=12)
    axes[0, 1].set_title('F1分数 vs 混合比例', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 鲁棒性分析（Clean vs Noisy性能差异）
    robustness = results_df['test_acc'] - results_df['noisy_test_acc']
    bars = axes[1, 0].bar(range(len(mix_ratios)), robustness, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('混合比例', fontsize=12)
    axes[1, 0].set_ylabel('性能下降 (Clean - Noisy)', fontsize=12)
    axes[1, 0].set_title('模型鲁棒性分析', fontsize=14, fontweight='bold')
    axes[1, 0].set_xticks(range(0, len(mix_ratios), 2))  # 只显示每隔一个标签
    axes[1, 0].set_xticklabels([f'{mix_ratios.iloc[i]:.2f}' for i in range(0, len(mix_ratios), 2)], rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 添加数值标签（只为显示的柱子添加标签）
    for i in range(0, len(robustness), 2):
        v = robustness.iloc[i]
        axes[1, 0].text(i, v + 0.005, f'{v:.3f}', 
                       ha='center', va='bottom', fontweight='bold')
    
    # 性能提升热力图
    metrics_matrix = results_df[['val_acc', 'test_acc', 'noisy_test_acc', 'val_f1', 'test_f1', 'noisy_test_f1']].T
    im = axes[1, 1].imshow(metrics_matrix, cmap='RdYlGn', aspect='auto')
    axes[1, 1].set_xticks(range(len(mix_ratios)))
    axes[1, 1].set_xticklabels([f'{r:.2f}' for r in mix_ratios], rotation=45, ha='right', fontsize=10)
    axes[1, 1].set_yticks(range(6))
    axes[1, 1].set_yticklabels(['Val Acc', 'Test Acc', 'Noisy Acc', 'Val F1', 'Test F1', 'Noisy F1'])
    axes[1, 1].set_xlabel('混合比例', fontsize=12)
    axes[1, 1].set_title('性能热力图', fontsize=14, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=axes[1, 1])
    cbar.set_label('性能分数', fontsize=11)
    
    # 在热力图上添加数值
    for i in range(metrics_matrix.shape[0]):
        for j in range(metrics_matrix.shape[1]):
            text = axes[1, 1].text(j, i, f'{metrics_matrix.iloc[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'comprehensive_comparison_analysis.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 最佳配置分析
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 找到各个指标的最佳配置
    best_val_acc_idx = results_df['val_acc'].idxmax()
    best_test_acc_idx = results_df['test_acc'].idxmax()
    best_noisy_acc_idx = results_df['noisy_test_acc'].idxmax()
    
    ax.plot(mix_ratios, results_df['test_acc'], 'b-o', linewidth=2, markersize=6, label='测试集准确率')
    ax.plot(mix_ratios, results_df['noisy_test_acc'], 'r-s', linewidth=2, markersize=6, label='OOD测试集准确率')
    
    # 标记最佳点
    ax.scatter(results_df.loc[best_test_acc_idx, 'mix_ratio'], 
              results_df.loc[best_test_acc_idx, 'test_acc'], 
              color='blue', s=200, marker='*', label=f'最佳Clean性能 (Ratio={results_df.loc[best_test_acc_idx, "mix_ratio"]:.2f})')
    ax.scatter(results_df.loc[best_noisy_acc_idx, 'mix_ratio'], 
              results_df.loc[best_noisy_acc_idx, 'noisy_test_acc'], 
              color='red', s=200, marker='*', label=f'最佳OOD性能 (Ratio={results_df.loc[best_noisy_acc_idx, "mix_ratio"]:.2f})')
    
    ax.set_xlabel('混合比例', fontsize=14)
    ax.set_ylabel('准确率', fontsize=14)
    ax.set_title('最佳混合比例分析', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    best_config_path = os.path.join(output_dir, 'best_configuration_analysis.png')
    plt.savefig(best_config_path, dpi=300, bbox_inches='tight')
    plt.close()

def summarize_all_experiments(results, output_dir, args, logger):
    """生成增强的实验总结报告"""
    if not results:
        logger.warning("没有实验结果可供总结")
        return

    results_df = pd.DataFrame(results)
    
    # 详细的统计分析
    logger.info("\n" + "="*80)
    logger.info(" " * 25 + "实验总结报告")
    logger.info("="*80)
    
    # 打印详细结果表格
    display_df = results_df.copy()
    float_cols = ['val_acc', 'val_f1', 'test_acc', 'test_f1', 'noisy_test_acc', 'noisy_test_f1']
    for col in float_cols:
        display_df[col] = display_df[col].map('{:.4f}'.format)
    
    logger.info("\n详细结果:")
    logger.info(display_df.to_string(index=False))
    
    # 统计分析
    logger.info("\n统计分析:")
    logger.info(f"最佳验证集准确率: {results_df['val_acc'].max():.4f} (Mix Ratio: {results_df.loc[results_df['val_acc'].idxmax(), 'mix_ratio']:.2f})")
    logger.info(f"最佳测试集准确率: {results_df['test_acc'].max():.4f} (Mix Ratio: {results_df.loc[results_df['test_acc'].idxmax(), 'mix_ratio']:.2f})")
    logger.info(f"最佳OOD准确率: {results_df['noisy_test_acc'].max():.4f} (Mix Ratio: {results_df.loc[results_df['noisy_test_acc'].idxmax(), 'mix_ratio']:.2f})")
    
    # 性能提升分析
    baseline_test_acc = results_df[results_df['mix_ratio'] == 0.0]['test_acc'].iloc[0]
    baseline_noisy_acc = results_df[results_df['mix_ratio'] == 0.0]['noisy_test_acc'].iloc[0]
    
    max_test_improvement = results_df['test_acc'].max() - baseline_test_acc
    max_noisy_improvement = results_df['noisy_test_acc'].max() - baseline_noisy_acc
    
    logger.info(f"\n性能提升分析 (相对于无增强baseline):")
    logger.info(f"测试集最大提升: {max_test_improvement:.4f} ({max_test_improvement/baseline_test_acc*100:.2f}%)")
    logger.info(f"OOD测试集最大提升: {max_noisy_improvement:.4f} ({max_noisy_improvement/baseline_noisy_acc*100:.2f}%)")
    
    logger.info("="*80)
    
    # 保存详细结果
    summary_csv_path = os.path.join(output_dir, 'comprehensive_experiment_summary.csv')
    results_df.to_csv(summary_csv_path, index=False)
    logger.info(f"详细结果已保存: {summary_csv_path}")
    
    # 保存统计摘要
    summary_stats = {
        'experiment_timestamp': datetime.now().isoformat(),
        'total_experiments': len(results),
        'mix_ratios_tested': results_df['mix_ratio'].tolist(),
        'best_configurations': {
            'best_val_acc': {'ratio': float(results_df.loc[results_df['val_acc'].idxmax(), 'mix_ratio']), 'value': float(results_df['val_acc'].max())},
            'best_test_acc': {'ratio': float(results_df.loc[results_df['test_acc'].idxmax(), 'mix_ratio']), 'value': float(results_df['test_acc'].max())},
            'best_noisy_acc': {'ratio': float(results_df.loc[results_df['noisy_test_acc'].idxmax(), 'mix_ratio']), 'value': float(results_df['noisy_test_acc'].max())}
        },
        'performance_improvements': {
            'test_acc_improvement': float(max_test_improvement),
            'noisy_acc_improvement': float(max_noisy_improvement)
        },
        'experimental_settings': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'batch_size': args.batch_size,
            'hidden_channels': args.hidden_channels,
            'noise_level': args.noise_level,
            'patience': args.patience
        }
    }
    
    summary_json_path = os.path.join(output_dir, 'experiment_summary_stats.json')
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    
    # 创建可视化
    create_final_comparison_plots(results_df, output_dir, args)
    logger.info("可视化图表已生成完成")

def main(args):
    """主执行流程"""
    # 设置随机种子
    set_random_seed(args.seed)
    
    # 设置日志
    logger = setup_logging(args.output_dir, 'data_augmentation_experiment')
    
    # 记录实验配置
    logger.info("实验配置:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")
    
    # 设置设备
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    original_dataset, generated_dataset = load_datasets(args.raw_data_dir, args.generated_data_path, logger)
    
    # 执行所有实验
    all_experiment_results = []
    total_experiments = len(args.mix_ratios)
    
    logger.info(f"\n开始执行 {total_experiments} 个实验...")
    overall_start_time = time.time()
    
    for i, ratio in enumerate(sorted(args.mix_ratios), 1):
        logger.info(f"\n进度: {i}/{total_experiments}")
        best_metrics = run_single_experiment(args, ratio, original_dataset, generated_dataset, device, logger)
        if best_metrics:
            best_metrics['mix_ratio'] = ratio
            all_experiment_results.append(best_metrics)
    
    total_time = time.time() - overall_start_time
    logger.info(f"\n所有实验完成! 总耗时: {total_time:.2f}s")
    
    # 生成总结报告
    summarize_all_experiments(all_experiment_results, args.output_dir, args, logger)
    
    logger.info(f"实验结果已保存至: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="增强版数据增强实验：使用混合数据训练GCN模型")
    
    # 路径参数
    parser.add_argument('--raw_data_dir', type=str, default='/data/wangzepeng/raw', help='原始数据目录')
    parser.add_argument('--generated_data_path', type=str, default='/data/wangzepeng/synthesis/generated_graphs.pt', help='生成数据路径')
    parser.add_argument('--output_dir', type=str, default='/home/wangzepeng/CGD_GNN/results/enhanced_aug_experiment', help='输出目录')

    # 模型和训练参数
    parser.add_argument('--device', type=int, default=0, help="CUDA设备ID")
    parser.add_argument('--epochs', type=int, default=200, help="最大训练轮数")
    parser.add_argument('--lr', type=float, default=0.005, help="学习率")
    parser.add_argument('--batch_size', type=int, default=128, help="批量大小")
    parser.add_argument('--hidden_channels', type=int, default=64, help="GCN隐藏层维度")
    parser.add_argument('--patience', type=int, default=20, help="早停耐心值")
    
    # 数据增强参数
    parser.add_argument('--mix_ratios', nargs='+', type=float, default=[0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20], 
                        help="混合比例列表")
    parser.add_argument('--noise_level', type=float, default=0.8, help="OOD噪声水平")
    parser.add_argument('--seed', type=int, default=42, help="随机种子")

    args = parser.parse_args()
    main(args)