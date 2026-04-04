"""Example usage of variant effect prediction functions."""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

from DGS.DL.Predict import (
    read_vcf,
    variants_to_intervals,
    VariantDataset,
    variant_effect_prediction,
    metric_predicted_effect,
    vep_centred_from_files
)

class SimpleConvNet(nn.Module):
    """Simple convolutional network for sequence prediction"""
    def __init__(self, input_channels=4, conv_channels=32, kernel_size=8):
        """Initialize `SimpleConvNet`."""
        super().__init__()
        self.conv = nn.Conv1d(input_channels, conv_channels, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(conv_channels, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        
    def forward(self, x):
        """Compute forward outputs for `SimpleConvNet`."""
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

def create_example_data():
    """Create example VCF and genome files"""
    # Create output directory
    output_dir = Path("example_data")
    output_dir.mkdir(exist_ok=True)
    
    # Create VCF file
    vcf_file = output_dir / "variants.vcf"
    with open(vcf_file, "w") as f:
        f.write("""##fileformat=VCFv4.2
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT
chr1\t1000\trs1\tA\tT\t100\tPASS\tAF=0.1\tGT
chr1\t2000\trs2\tG\tC\t100\tPASS\tAF=0.2\tGT
chr2\t1500\trs3\tAG\tA\t100\tPASS\tAF=0.3\tGT
""")
    
    # Create genome file
    genome_file = output_dir / "genome.fa"
    with open(genome_file, "w") as f:
        f.write(""">chr1
ATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCG
>chr2
GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA
""")
    
    return vcf_file, genome_file

def main():
    # 设置随机种子
    """Run the end-to-end usage example."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建示例数据和模型
    vcf_file, genome_file = create_example_data()
    model = SimpleConvNet()
    
    print("\n1. 读取变体信息")
    variants = read_vcf(vcf_file)
    print("变体数量:", len(variants))
    print("\n变体信息示例:")
    print(variants.head())
    
    print("\n2. 转换为区间")
    intervals = variants_to_intervals(variants, seq_len=100)
    print("区间信息示例:")
    print(intervals.head())
    
    print("\n3. 创建变体数据集")
    dataset = VariantDataset(genome_file, variants, target_len=100)
    print("数据集大小:", len(dataset))
    
    # 获取第一个变体的序列
    seq_ref, seq_alt = dataset[0]
    print("\n参考序列长度:", len(seq_ref))
    print("变异序列长度:", len(seq_alt))
    
    print("\n4. 预测变体效应")
    p_ref, p_alt = variant_effect_prediction(model, seq_ref, seq_alt)
    print("预测形状:")
    print("参考序列预测:", p_ref.shape)
    print("变异序列预测:", p_alt.shape)
    
    print("\n5. 计算效应分数")
    metrics = ['diff', 'ratio', 'log_ratio', 'max', 'min']
    print("\n不同效应度量方法的结果:")
    for metric in metrics:
        p_eff = metric_predicted_effect(p_ref, p_alt, metric_func=metric)
        print(f"{metric}: {p_eff}")
    
    print("\n6. 从文件进行变体效应预测")
    try:
        result_df = vep_centred_from_files(
            model,
            genome_file,
            vcf_file,
            target_len=100,
            metric_func='diff',
            return_df=True
        )
        print("\n预测结果:")
        print(result_df)
        
    except Exception as e:
        print(f"运行 vep_centred_from_files 时出错: {str(e)}")

if __name__ == "__main__":
    main() 
