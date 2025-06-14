# ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval

PyTorch implementation of **ConText-CIR** from the paper "ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval"

ConText-CIR is trained with a novel Text Concept-Consistency loss that encourages better alignment between noun phrases in text and their corresponding image regions.

## 🔧 Installation

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA 11.8+

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/ConText-CIR.git
cd ConText-CIR

# Install dependencies
pip install -r requirements.txt
```

## 📁 Data Preparation

### CIRR Dataset
1. Download CIRR dataset from [here](https://github.com/Cuberick-Orion/CIRR)
2. Extract to `data/cirr/`

### LaSCo Dataset
1. Download from [here] (https://github.com/levymsn/LaSCo)
2. PLace in `data/lasco/`

### CIRR-Rewritten Dataset (Optional)
1. Download rewritten captions from our release
2. Place in `data/cirr/rewritten_captions/`

### Hotel-CIR Dataset (Optional)
1. Download Hotel-CIR from our release
2. Extract to `data/hotels/`


### Configuration
Create a `config.yaml` file:
```yaml
model_dir: ./checkpoints
cirr_data_path: ./data/cirr
hotel_data_path: ./data/hotels
lasco_data_path: ./data/lasco
```

## 🏃 Training

### Basic Training
```bash
# Train with CIRR dataset on single GPU
python Train.py --datasets cirr \
                --backbone_size B \
                --train_batch_size 256 \
                --learning_rate 1e-5

# Train with multiple datasets on multiple GPUs
python Train.py --datasets cirr,cirr_r,hotels \
                --backbone_size H \
                --devices 4 \
                --train_batch_size 64 \
                --lambda_cc 0.08
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | Required | Comma-separated list: cirr, cirr_r, hotels, lasco |
| `--backbone_size` | B | CLIP model size: B, L, or H |
| `--lambda_cc` | 0.08 | Weight for concept-consistency loss |
| `--epsilon_cc` | 0.05 | Slack variable for CC loss |
| `--max_nps` | 10 | Max noun phrases per text |
| `--train_batch_size` | 256 | Training batch size |
| `--learning_rate` | 1e-5 | Learning rate |
| `--devices` | 1 | Number of GPUs |
| `--approx_steps` | 35000 | Approximate training steps |

### Other Training Options
```bash
# Resume from checkpoint
python Train.py --datasets cirr \
                --backbone_size H \
                --reload \
                --output_dir ./checkpoints/experiment1
```

## 📈 Evaluation
We provide utilities to produce submissions to the CIRR and CIRCO testing servers. 

## 📝 Citation

If you find this code useful for your research, please cite:

```bibtex
@article{xing2025context,
  title={ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval},
  author={Xing, Eric and Kolouju, Pranavi and Pless, Robert and Stylianou, Abby and Jacobs, Nathan},
  journal={Computer Vision and Paattern Recognition},
  year={2025}
}
```