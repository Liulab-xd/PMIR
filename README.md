# PMIR

Polarization-guided underwater image restoration with physical modeling and wavelet detail refinement.

PMIR jointly restores RGB observations captured at 0°, 45°, 90°, and 135°. The network combines shared polarization-conditioned Retinex preprocessing, a multiscale encoder–decoder, optical-depth and background Stokes estimation, physical inversion, and high-frequency residual refinement.

## Installation

Use Python 3.10 or later with a PyTorch installation suitable for your hardware.

```bash
python -m pip install -e .
```

Alternatively, install the dependencies with `python -m pip install -r requirements.txt` and run from the repository root.

## Repository layout

```text
PMIR/
├── pmir/
│   ├── cli/
│   │   └── train.py
│   ├── data/
│   │   ├── augmentation.py
│   │   ├── dataset.py
│   │   ├── image_io.py
│   │   └── polarization.py
│   ├── engine/
│   │   └── trainer.py
│   ├── losses/
│   │   └── pol_loss.py
│   ├── models/
│   │   ├── pmir.py
│   │   └── modules/
│   ├── config.py
│   └── __main__.py
├── docs/
│   ├── architecture.md
│   └── data.md
├── pyproject.toml
├── requirements.txt
└── train.py
```

The model, data pipeline, objective, command-line interface, and training engine are maintained separately. Package directories include `__init__.py` files.

## Data preparation

Prepare four paired RGB BMP observations per scene for both scattered and clear images:

```text
data/
├── scatter/
│   ├── scatter_0/0.bmp
│   ├── scatter_45/45.bmp
│   ├── scatter_90/90.bmp
│   └── scatter_135/135.bmp
└── raw/
    ├── raw_0/0.bmp
    ├── raw_45/45.bmp
    ├── raw_90/90.bmp
    └── raw_135/135.bmp
```

See [Data format](docs/data.md) for pairing, tensor layout, and augmentation details.

## Training

> **Configuration notice:** Some parameter values in this public release are provided as defaults and differ from those used in the actual training experiments. Before training, review and adjust the configuration for your dataset, hardware, and experimental setup, including the number of epochs, batch size, learning rate, feature width, and loss weights. Set supported training options through the command-line interface and edit loss weights in `pmir/config.py`. Using the default configuration does not guarantee reproduction of experimental results. When loading an existing checkpoint, model architecture settings must match those used to train that checkpoint.

```bash
python train.py --scatter-root data/scatter --raw-root data/raw --output-dir outputs/experiment_01
```

Equivalent entry points are `python -m pmir` and, after installation, `pmir-train`. Run `python train.py --help` to inspect all options.

Training uses every complete paired sample in the supplied dataset. The defaults are 350 epochs, batch size 4, image size 256, feature width 48, AdamW with learning rate 0.0002 and weight decay 0.0001, cosine learning-rate scheduling, and gradient clipping at 5.0. Polarization augmentation and Retinex preprocessing are enabled by default; use `--no-augment` or `--no-retinex` to disable either option.

An output directory must be empty when a run starts. Relative paths resolve from the current working directory. Each run writes:

```text
outputs/experiment_01/
├── config.json
├── metrics.csv
└── checkpoints/
    ├── last_model.pth
    └── epoch_0050.pth
```

`last_model.pth` is updated after every completed epoch. Numbered checkpoints are saved every 50 epochs by default and at the final epoch; change this with `--save-interval`. Each checkpoint contains the epoch, model state, optimizer state, scheduler state, training loss, and run configuration. The recorded losses are training losses.

## Model interface

```python
import torch

from pmir.models import PMIR

model = PMIR(feature_channels=48, use_pol_retinex=True)
angle_input = torch.rand(1, 12, 256, 256)
dop = torch.rand(1, 1, 256, 256)
outputs = model(angle_input, dop)
restored_angles = outputs[0]
```

The 12 input and output channels are grouped as RGB at 0°, 45°, 90°, and 135°. In actual use, compute the DoLP guide from the four observations with `pmir.data.polarization.compute_dop`; the random tensors above illustrate the interface only. See [Architecture](docs/architecture.md) for the complete output contract and training objective.

Model parameters are stored under the `model` key in checkpoints. Match `feature_channels` and `use_pol_retinex` to the training configuration when loading a state dictionary.
