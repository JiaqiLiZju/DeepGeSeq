"""Example usage of the Evaluator class."""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pathlib import Path

from DGS.DL.Evaluator import Evaluator

def create_dummy_data():
    """创建示例数据"""
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建特征和标签
    n_samples = 1000
    n_features = 10
    n_classes = 3
    
    X = torch.randn(n_samples, n_features)
    y_binary = torch.randint(0, 2, (n_samples,))
    y_multi = torch.randint(0, n_classes, (n_samples,))
    y_reg = torch.randn(n_samples, 1)
    
    return X, y_binary, y_multi, y_reg

def create_model(input_dim, output_dim):
    """创建简单模型"""
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim)
    )

def custom_f2_score(y_true, y_pred, beta=2):
    """自定义F2分数计算"""
    y_pred = (y_pred > 0.5).astype(np.float32)
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positives = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-7)
    return float(f2)

def main():
    # 创建输出目录
    """Run the end-to-end usage example."""
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    # 准备数据
    X, y_binary, y_multi, y_reg = create_dummy_data()
    
    # 1. 二分类评估示例
    print("\n1. Binary Classification Example:")
    binary_model = create_model(X.shape[1], 1)
    binary_dataset = TensorDataset(X, y_binary)
    binary_loader = DataLoader(binary_dataset, batch_size=32)
    
    binary_evaluator = Evaluator(
        model=binary_model,
        test_loader=binary_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        task_type="binary_classification",
        output_dir=output_dir / "binary"
    )
    
    # 添加自定义指标
    binary_evaluator.add_custom_metric("f2_score", custom_f2_score)
    
    # 评估并打印结果
    metrics, error_analysis = binary_evaluator.evaluate()
    print("\nMetrics:")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"F1 Score: {metrics.f1:.3f}")
    print(f"AUROC: {metrics.auroc['binary']:.3f}")
    
    print("\nError Analysis:")
    print(f"False Positives: {len(error_analysis.false_positives)}")
    print(f"False Negatives: {len(error_analysis.false_negatives)}")
    
    # 2. 多分类评估示例
    print("\n2. Multiclass Classification Example:")
    multi_model = create_model(X.shape[1], 3)
    multi_dataset = TensorDataset(X, y_multi)
    multi_loader = DataLoader(multi_dataset, batch_size=32)
    
    multi_evaluator = Evaluator(
        model=multi_model,
        test_loader=multi_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        task_type="multiclass_classification",
        num_classes=3,
        output_dir=output_dir / "multiclass"
    )
    
    metrics, error_analysis = multi_evaluator.evaluate()
    print("\nMetrics:")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Macro AUROC: {metrics.auroc['macro']:.3f}")
    print(f"Micro AUROC: {metrics.auroc['micro']:.3f}")
    
    # 3. 回归评估示例
    print("\n3. Regression Example:")
    reg_model = create_model(X.shape[1], 1)
    reg_dataset = TensorDataset(X, y_reg)
    reg_loader = DataLoader(reg_dataset, batch_size=32)
    
    reg_evaluator = Evaluator(
        model=reg_model,
        test_loader=reg_loader,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        task_type="regression",
        output_dir=output_dir / "regression"
    )
    
    metrics = reg_evaluator.evaluate()
    print("\nMetrics:")
    print(f"MSE: {metrics.mse:.3f}")
    print(f"RMSE: {metrics.rmse:.3f}")
    print(f"R2 Score: {metrics.r2:.3f}")
    print(f"Pearson Correlation: {metrics.pearson_r:.3f}")

if __name__ == "__main__":
    main() 
