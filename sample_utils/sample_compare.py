# python3 /home/wangzepeng/CGD_GNN/sample_utils/sample_compare.py
import sys
import os
import argparse
from datetime import datetime
# 动态添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from src.models.gaussian_diffusion import GuidedGaussianDiffusion
from src.models.Unet import ConditionalDiffusionUNet
from src.models.classifier import UNetClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

def main(args):
    # -- 1. 设置和配置 --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建基于时间戳的唯一输出目录
    eval_output_dir = os.path.join(args.output_dir, f"scale_eval_{timestamp}")
    os.makedirs(eval_output_dir, exist_ok=True)

    print(f"评估结果将保存至: {eval_output_dir}")

    # -- 2. 加载模型 --
    # 加载U-Net模型
    print(f"加载U-Net模型: {args.unet_path}")
    unet = ConditionalDiffusionUNet(num_classes=args.num_classes, time_dim=128, label_dim=64).to(device)
    try:
        unet.load_state_dict(torch.load(args.unet_path, map_location=device))
        unet.eval()
    except Exception as e:
        print(f"加载U-Net模型失败: {e}"); sys.exit(1)

    # 加载引导分类器 (如果需要)
    guidance_classifier = None
    if any(s > 0 for s in args.classifier_scales):
        print(f"加载引导分类器: {args.guidance_classifier_path}")
        try:
            guidance_classifier = UNetClassifier(args.num_classes).to(device)
            guidance_classifier.load_state_dict(torch.load(args.guidance_classifier_path, map_location=device))
            guidance_classifier.eval()
        except Exception as e:
            print(f"加载引导分类器失败: {e}"); sys.exit(1)
            
    # 加载评估分类器
    print(f"加载评估分类器: {args.eval_classifier_path}")
    eval_classifier = UNetClassifier(args.num_classes).to(device)
    try:
        eval_classifier.load_state_dict(torch.load(args.eval_classifier_path, map_location=device))
        eval_classifier.eval()
    except Exception as e:
        print(f"加载评估分类器失败: {e}"); sys.exit(1)

    # -- 3. 循环评估每个引导强度 --
    all_results = []
    
    for scale in args.classifier_scales:
        print(f"\n===== 评估引导强度: {scale} =====")
        
        # 初始化扩散模型
        diffusion = GuidedGaussianDiffusion(
            num_steps=args.diffusion_steps,
            classifier_scale=scale
        )
        
        for class_idx in range(args.num_classes):
            class_predictions = []
            
            pbar = tqdm(range(0, args.samples_per_class, args.batch_size), 
                        desc=f"生成类别 {class_idx} (scale={scale})")
            
            for _ in pbar:
                current_batch_size = min(args.batch_size, args.samples_per_class - len(class_predictions))
                if current_batch_size <= 0: break
                
                target_labels = torch.full((current_batch_size,), class_idx, device=device).long()
                
                with torch.no_grad():
                    generated = diffusion.p_sample_loop(
                        model=unet,
                        classifier=guidance_classifier,
                        shape=(current_batch_size, 1, 24, 50),
                        batch_labels=target_labels,
                        progress=False
                    )
                    
                    # 使用评估分类器评估生成的样本
                    dummy_timesteps = torch.zeros(current_batch_size, device=device).long()
                    logits = eval_classifier(generated, dummy_timesteps)
                    batch_predictions = logits.argmax(dim=1)
                    class_predictions.extend(batch_predictions.cpu().tolist())
            
            # 计算当前类别的生成准确率
            success_count = sum(p == class_idx for p in class_predictions)
            accuracy = success_count / len(class_predictions) if class_predictions else 0
            
            all_results.append({
                'Classifier Scale': scale,
                'Class': f'Class {class_idx}',
                'Success Rate': accuracy
            })
            print(f"类别 {class_idx} (scale={scale}) -> 成功率: {accuracy:.2%}")

    # -- 4. 生成可视化图表和报告 --
    results_df = pd.DataFrame(all_results)
    
    # 绘制对比图
    plt.figure(figsize=(16, 9))
    sns.barplot(data=results_df, x='Class', y='Success Rate', hue='Classifier Scale', palette="ch:s=.25,rot=-.25")
    
    plt.title('Success Rate by Class and Classifier Scale', fontsize=16)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Success Rate', fontsize=12)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Classifier Scale')
    plt.tight_layout()
    
    plot_path = os.path.join(eval_output_dir, 'success_rate_by_scale.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"\n对比图已保存至: {plot_path}")

    # 保存结果到CSV
    csv_path = os.path.join(eval_output_dir, 'success_rate_by_scale.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"详细数据已保存至: {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估不同分类器引导强度下的样本生成成功率")
    
    # 路径参数
    parser.add_argument('--unet_path', type=str, default='/data/wangzepeng/models/unet_sample/best_unet.pth', 
                        help='U-Net检查点文件路径 (.pth)')
    parser.add_argument('--guidance_classifier_path', type=str, default='/data/wangzepeng/models/classifier/best_noisy_classifier.pth',
                        help='用于引导生成的(noisy)分类器模型路径')
    parser.add_argument('--eval_classifier_path', type=str, default='/data/wangzepeng/models/classifier/best_eval_classifier.pth',
                        help='用于评估生成质量的(clean)分类器模型路径')
    parser.add_argument('--output_dir', type=str, default='/home/wangzepeng/CGD_GNN/results/scale_evaluation',
                        help='保存评估结果 (图表、报告) 的根目录')

    # 生成和模型参数
    parser.add_argument('--num_classes', type=int, default=7, help='数据集的类别数')
    parser.add_argument('--samples_per_class', type=int, default=100, help='每个类别/强度的组合要生成的样本数量')
    parser.add_argument('--batch_size', type=int, default=50, help='生成样本时的批处理大小')
    
    # 扩散模型参数
    parser.add_argument('--diffusion_steps', type=int, default=64, help='扩散/采样过程的总步数')
    parser.add_argument('--classifier_scales', nargs='+', type=float, default=[0.0, 1.0, 3.0, 5.0, 10.0],
                        help='要评估的一系列分类器引导强度值')
    
    args = parser.parse_args()
    main(args)