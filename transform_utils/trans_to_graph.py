# python3 /home/wangzepeng/CGD_GNN/transform_utils/trans_to_graph.py
import sys
import os
import argparse
import torch

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设 src 目录在当前目录的上一级
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# from data.pyg_dataToGraph import DataToGraph # 旧的导入
from src.utils.pyg_dataToGraph import DataToGraph # 新的导入，假设项目根目录在PYTHONPATH中

def main(args):
    # -- 1. 加载数据 --
    print(f"加载生成的张量数据: {args.generated_data_path}")
    if not os.path.exists(args.generated_data_path):
        print(f"错误: 找不到生成的数据文件: {args.generated_data_path}")
        sys.exit(1)
    generated_data = torch.load(args.generated_data_path)

    print(f"加载原始图结构模板: {args.raw_data_dir}")
    dataset = DataToGraph(
        raw_data_path=args.raw_data_dir,
        dataset_name='TFF.mat'
    )
    print(f"原始图模板: {dataset[0]}")

    gen_x = generated_data.get('samples')
    gen_label = generated_data.get('target_labels')

    if gen_x is None or gen_label is None:
        print("错误: 无法在生成的数据中找到 'samples' 或 'target_labels' 字段。")
        sys.exit(1)
        
    print(f"原始数据集大小: {len(dataset)}")
    print(f"生成数据集大小: {len(gen_x)}")
    print(f"原始数据 x 形状: {dataset[0].x.shape}")
    print(f"生成数据 x 形状: {gen_x[0].shape}")

    # -- 2. 转换数据 --
    new_dataset = []
    template_graph = dataset[0] # 使用第一个图作为结构模板

    for i in range(len(gen_x)):
        new_graph = template_graph.clone()
        
        # 确保节点特征维度匹配
        if gen_x[i].dim() == 3 and gen_x[i].shape[0] == 1: # [1, 24, 50]
            new_graph.x = gen_x[i].squeeze(0) # -> [24, 50]
        else:
            new_graph.x = gen_x[i]

        # 确保标签格式正确 - 转换为 [y] 格式的张量
        label = gen_label[i]
        if isinstance(label, torch.Tensor):
            new_graph.y = label.clone().detach().view(1)
        else: # 如果是python的 int 或 float
            new_graph.y = torch.tensor([label], dtype=torch.long)
            
        new_dataset.append(new_graph)

    print(f"\n转换后的第一个图示例: {new_dataset[0]}")

    # -- 3. 保存结果 --
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    torch.save(new_dataset, args.output_path)
    print(f"\n✅ 生成的图数据集已保存至: {args.output_path}")
    
    # 打印一些统计信息
    print("\n📊 转换总结:")
    print(f"   - 新数据集大小: {len(new_dataset)}")
    print(f"   - 样本x形状: {new_dataset[0].x.shape}")
    print(f"   - 样本y形状: {new_dataset[0].y.shape}")
    print(f"   - 样本edge_index形状: {new_dataset[0].edge_index.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将生成的Tensor数据转换为PyG图结构数据")

    parser.add_argument('--generated_data_path', type=str, 
                        default='/data/wangzepeng/synthesis/generated_data.pt', 
                        help='生成的张量数据文件路径 (.pt)')
    parser.add_argument('--raw_data_dir', type=str, 
                        default='/data/wangzepeng/raw', 
                        help='包含TFF.mat的原始数据目录')
    parser.add_argument('--output_path', type=str, 
                        default='/data/wangzepeng/synthesis/generated_graphs.pt', 
                        help='转换后的图数据集的输出路径 (.pt)')

    args = parser.parse_args()
    main(args)