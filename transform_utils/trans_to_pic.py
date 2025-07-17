# python3 /home/wangzepeng/Augmentation/transform_utils/trans_to_pic.py
import sys
import os
import argparse
import torch

# # 获取当前文件的目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # 假设 src 目录在当前目录的上一级
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from data.pyg_dataToGraph import DataToGraph # 旧的导入
from src.utils.pyg_dataToGraph import DataToGraph # 新的导入

def main(args):
    # -- 1. 加载和处理数据 --
    print(f"从 {args.raw_data_dir} 加载原始 .mat 数据...")
dataset = DataToGraph(
        raw_data_path=args.raw_data_dir,
        dataset_name='TFF.mat'
    )
num_classes = dataset.num_classes
    print(f"原始图数据示例: {dataset[0]}")

    x0, labels = [], []
for data in dataset:
    x0.append(data.x)
    labels.append(data.y)

# 将列表转换为张量
    x0 = torch.stack(x0).unsqueeze(1)  # -> [N, 1, 24, 50]
    labels = torch.stack(labels)      # -> [N, 1]
    
    # 确保标签维度正确
    if labels.dim() > 1 and labels.shape[-1] == 1:
        labels = labels.squeeze(-1) # -> [N]

    print(f"\n类别数: {num_classes}")
    print(f"转换后的 X0 shape: {x0.shape}")
    print(f"转换后的 Labels shape: {labels.shape}")

    # -- 2. 保存结果 --
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save({
        'samples': x0,
        'labels': labels
    }, args.output_path)
    
    print(f"\n✅ 已将转换后的张量数据保存至: {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="将PyG图数据转换为Tensor(图片)格式")

    parser.add_argument('--raw_data_dir', type=str, 
                        default='/data/wangzepeng/raw', 
                        help='包含TFF.mat的原始数据目录')
    parser.add_argument('--output_path', type=str, 
                        default='/data/wangzepeng/raw/TFF_tensors.pt', 
                        help='转换后的张量数据的输出路径 (.pt)')

    args = parser.parse_args()
    main(args)