# python3 /home/wangzepeng/CGD_GNN/sample_utils/sample.py
import sys
import os
import argparse
from datetime import datetime
# 动态添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import numpy as np
from src.models.gaussian_diffusion import GuidedGaussianDiffusion
from src.models.Unet import ConditionalDiffusionUNet
from src.models.classifier import UNetClassifier
from tqdm import tqdm

def main(args):
    # -- 1. 设置和配置 --
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # -- 2. 加载模型 --
    # 加载U-Net模型
    print(f"加载U-Net模型: {args.unet_path}")
    unet = ConditionalDiffusionUNet(num_classes=args.num_classes, time_dim=128, label_dim=64).to(device)
    try:
        unet.load_state_dict(torch.load(args.unet_path, map_location=device))
        unet.eval()
        print(f"成功加载U-Net模型: {os.path.basename(args.unet_path)}")
    except Exception as e:
        print(f"加载U-Net模型失败: {e}")
        sys.exit(1)

    # 加载引导分类器 (如果需要)
    guidance_classifier = None
    if args.classifier_scale > 0:
        print(f"加载引导分类器: {args.classifier_path}")
        try:
            guidance_classifier = UNetClassifier(args.num_classes).to(device)
            guidance_classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
            guidance_classifier.eval()
            print(f"成功加载引导分类器: {args.classifier_path}")
        except Exception as e:
            print(f"加载引导分类器失败: {e}\n错误: 分类器引导模式需要一个有效的引导分类器模型。")
            sys.exit(1)
            
    # 加载评估分类器
    print(f"加载评估分类器: {args.eval_classifier_path}")
    eval_classifier = UNetClassifier(args.num_classes).to(device)
    try:
        eval_classifier.load_state_dict(torch.load(args.classifier_path, map_location=device))
        eval_classifier.load_state_dict(torch.load(args.eval_classifier_path, map_location=device))
        eval_classifier.eval()
        print(f"成功加载评估分类器: {os.path.basename(args.eval_classifier_path)}")
    except Exception as e:
        print(f"加载评估分类器失败: {e}")
        sys.exit(1)

    # 初始化扩散模型
    diffusion = GuidedGaussianDiffusion(
        num_steps=args.diffusion_steps,
        classifier_scale=args.classifier_scale
    )

    # -- 3. 生成并过滤高质量样本 --
    all_hq_samples = []
    all_hq_labels = []
    
    print("\n开始生成并过滤高质量样本...")
    # 为每个类别生成样本
    for class_idx in range(args.num_classes):
        hq_samples_for_class = []
        total_generated_count = 0
        
        with tqdm(total=args.samples_per_class, desc=f"收集类别 {class_idx}") as pbar:
            while pbar.n < args.samples_per_class:
            
                target_labels = torch.full((args.batch_size,), class_idx, device=device).long()
            
                with torch.no_grad():
                    generated_batch = diffusion.p_sample_loop(
                        model=unet,
                        classifier=guidance_classifier,
                        shape=(args.batch_size, 1, 24, 50),
                        batch_labels=target_labels,
                        progress=False
                    )
                    total_generated_count += args.batch_size

                    # 使用评估分类器进行过滤
                    dummy_timesteps = torch.zeros(args.batch_size, device=device).long()
                    logits = eval_classifier(generated_batch, dummy_timesteps)
                    predicted_labels = logits.argmax(dim=1)

                    # 筛选出高质量的样本
                    mask = (predicted_labels == class_idx)
                    passed_samples = generated_batch[mask]
                    
                    if passed_samples.numel() > 0:
                        # 计算还需要多少样本，并确保不会超额添加
                        needed = args.samples_per_class - pbar.n
                        samples_to_add = passed_samples[:needed]
                        hq_samples_for_class.append(samples_to_add.cpu())
                        pbar.update(samples_to_add.shape[0])

                # 更新调试信息
                acceptance_rate = (pbar.n / total_generated_count) * 100 if total_generated_count > 0 else 0
                pbar.set_postfix({
                    '已生成': total_generated_count,
                    '接受率': f'{acceptance_rate:.2f}%'
                })
        
        # 将当前类别收集到的高质量样本整合
        if hq_samples_for_class:
            class_samples_tensor = torch.cat(hq_samples_for_class, dim=0)
            class_labels_tensor = torch.full((class_samples_tensor.shape[0],), class_idx, dtype=torch.long)
            all_hq_samples.append(class_samples_tensor)
            all_hq_labels.append(class_labels_tensor)

    # 将所有列表中的张量合并成最终的大张量
    final_samples = torch.cat(all_hq_samples, dim=0)
    final_labels = torch.cat(all_hq_labels, dim=0)

    print("\n样本生成和过滤完成。")
    print(f"总共获得高质量样本数量: {final_samples.shape[0]}")
    print(f"最终样本张量形状: {final_samples.shape}")
    print(f"最终标签张量形状: {final_labels.shape}")
    
    # -- 4. 保存生成的数据 --
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save({
        'samples': final_samples,
        'target_labels': final_labels
    }, args.output_path)
    
    print(f"\n✅ 高质量生成数据已保存至: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用引导扩散模型生成并通过评估分类器过滤高质量样本")
    
    # 路径参数
    parser.add_argument('--unet_path', type=str, default='/data/wangzepeng/models/unet_sample/best_unet.pth', 
                        help='U-Net检查点文件路径 (.pth)')
    parser.add_argument('--classifier_path', type=str, default='/data/wangzepeng/models/classifier/best_noisy_classifier.pth',
                        help='用于引导生成的分类器模型路径')
    parser.add_argument('--eval_classifier_path', type=str, default='/data/wangzepeng/models/classifier/best_eval_classifier.pth',
                        help='用于过滤样本的评估分类器模型路径')
    parser.add_argument('--output_path', type=str, default='/data/wangzepeng/synthesis/generated_data.pt',
                        help='保存生成的样本和标签的文件路径 (.pt)')

    # 生成和模型参数
    parser.add_argument('--num_classes', type=int, default=7, help='数据集的类别数')
    parser.add_argument('--samples_per_class', type=int, default=1000, help='每个类别要收集的高质量样本数量')
    parser.add_argument('--batch_size', type=int, default=50, help='每批次生成样本的数量')
    
    # 扩散模型参数
    parser.add_argument('--diffusion_steps', type=int, default=64, help='扩散/采样过程的总步数')
    parser.add_argument('--classifier_scale', type=float, default=3.0, help='分类器引导的强度。设置为0则为无引导生成。')
    
    args = parser.parse_args()
    main(args)