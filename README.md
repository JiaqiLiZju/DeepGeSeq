# DeepGeSeq 

<div align=left><img src="./Figures/Logo.png" width="80px"></img></div>

DeepGeSeq is a systematic and easy-to-use deep learning toolkit for genomics data analysis. It provides comprehensive support for modern deep learning architectures and analysis pipelines in genomics research.

<div align=center><img src="./Figures/DeepGeSeq.png" width="450px"></img></div>

## Key Features

### 1. Model Architectures
- **Modern Deep Learning Support**:
  - Residual Networks (ResNet)
  - Attention Mechanisms (CBAM)
  - Transformers
  - Customizable architectures
- **Pre-implemented Models**:
  - Re-implementation of published models
  - Easy model customization
  - Automatic architecture search

### 2. Training & Evaluation
- **Flexible Training Pipeline**:
  - GPU acceleration support
  - Early stopping
  - Checkpoint management
  - TensorBoard integration
- **Comprehensive Evaluation**:
  - Multiple metrics (AUC, PR, F1, etc.)
  - Cross-validation support
  - Performance visualization

### 3. Model Applications
- **Motif Analysis**:
  - Motif enrichment analysis
  - Integration with MEME Suite
  - HOMER motif search support
- **Feature Attribution**:
  - Gradient-based attribution
  - Integrated gradients
  - DeepLIFT support
- **Variant Effect Prediction**:
  - VCF file support
  - Variant impact scoring
  - Batch prediction

### 4. Data Processing
- **Multiple Format Support**:
  - FASTA/FASTQ sequences
  - BED intervals
  - BigWig signals
  - VCF variants
- **Efficient Processing**:
  - Parallel data loading
  - Memory-efficient processing
  - Strand-aware analysis

## Installation

### Requirements
- Python ≥ 3.7
- PyTorch ≥ 1.10.1
- CUDA support (optional, for GPU acceleration)

### Dependencies
```bash
# Core dependencies
numpy
pandas>=0.21
matplotlib==3.0.*
h5py==2.10.0  # Note: h5py > 2.10 may return b'strings' when reading h5file
tqdm
scikit-learn>=0.21.2
torch>=1.10.1  # Required for tensorboard and ModifyOutputHook
tensorboard==2.7.0
captum==0.5.0
networkx
pillow

# Optional external tools
meme==5.4.1  # For motif database comparison
homer2      # For motif search in activated seqlets
```

### Installation Methods

1. **From Source**:
```bash
# Clone repository
git clone https://github.com/JiaqiLiZju/DeepGeSeq.git

# Install
cd DeepGeSeq
python setup.py install
```

2. **Development Installation**:
```bash
# For development and testing
python setup.py develop
```

## Quick Start

### 1. Generate Configuration
```bash
# Generate minimal example config
nvtk config --example minimal --output config.json

# Generate full example config
nvtk config --example full --output config.json
```

### 2. Basic Training
```bash
# Train with default settings
nvtk train --config train_config.json

# Train with specific GPU
nvtk train --config train_config.json --gpu 0 --seed 42
```

### 3. Model Evaluation
```bash
# Basic evaluation
nvtk evaluate --config eval_config.json

# Evaluation with specific metrics
nvtk evaluate --config eval_config.json --metrics auc,pr,f1
```

### 4. Model Interpretation
```bash
# Generate motif explanations
nvtk explain --config explain_config.json

# Predict variant effects
nvtk predict --config predict_config.json
```

## Configuration Examples

### Minimal Configuration
```json
{
  "data": {
    "genome_path": "data/genome.fa",
    "intervals_path": "data/intervals.bed",
    "target_tasks": [
      {
        "task_name": "binding",
        "file_path": "data/chip.bw",
        "file_type": "bigwig"
      }
    ]
  },
  "model": {
    "type": "CNN",
    "args": {"output_size": 1}
  }
}
```

### Advanced Configuration
```json
{
  "data": {
    "genome_path": "data/genome.fa",
    "intervals_path": "data/intervals.bed",
    "target_tasks": [
      {
        "task_name": "binding",
        "file_path": "data/chip.bw",
        "file_type": "bigwig"
      }
    ],
    "batch_size": 64,
    "strand_aware": true,
    "sequence_length": 1000
  },
  "model": {
    "type": "ResNet",
    "args": {
      "num_layers": 34,
      "hidden_size": 256,
      "dropout": 0.1
    }
  },
  "train": {
    "optimizer": {
      "type": "Adam",
      "params": {
        "lr": 0.001,
        "weight_decay": 1e-6
      }
    },
    "max_epochs": 100,
    "patience": 10,
    "use_tensorboard": true
  }
}
```

## Advanced Usage

### Define your data
```python
from DeepGeSeq.Dataset import SeqDataset
from DeepGeSeq.Data import create_dataloader
class MyData(SeqDataset):
    def __init__(self,
        intervals: Interval,
        genome: Genome,
        targets: Target,
        strand_aware: bool = True
    ):
        super().__init__(intervals, genome, strand_aware)
        # add your own labels here
        self.labels = labels

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a sequence and its labels by index."""
        seq = self.seqs[idx]
        label = self.labels[idx]
        return seq.to_onehot(), label

my_dataset = MyData(intervals, genome, labels)
my_dataloader = create_dataloader(my_dataset, batch_size=32, shuffle=True)
```

### Custom Model Development
```python
import torch
from DeepGeSeq.Model import BaseModel

class MyModel(BaseModel):
    def __init__(self, input_size=1000, output_size=1):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 64, 3)
        self.fc = torch.nn.Linear(64 * (input_size-2), output_size)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### Custom Training Loop
```python
from DeepGeSeq.DL import Trainer

# Initialize trainer with custom settings
trainer = Trainer(
    model=my_model,
    criterion=torch.nn.BCELoss(),
    optimizer=torch.optim.Adam(my_model.parameters()),
    device=device,
    checkpoint_dir="checkpoints",
    use_tensorboard=True
)

# Train model
trainer.train(
    my_dataloader,
    my_dataloader,
    epochs=100,
    early_stopping=True
)
```

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## Citation

If you use DeepGeSeq in your research, please cite:
```bibtex
@article{li2024deepgeseq,
  title={DeepGeSeq: A systematic deep learning toolkit for genomics},
  author={Li, Jiaqi and Wu, Hanyu and others},
  journal={},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Issues: Please use the [GitHub issue tracker](https://github.com/JiaqiLiZju/DeepGeSeq/issues)
- Email: jiaqili@zju.edu.cn

## News
- 2020.03: DeepGeSeq is quite unstable under activate development.
- 2024.10: updating CaseStudies in Manuscript
- 2025.02: updating Documents, Tutorials
