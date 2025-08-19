## NeuroPIGAN

Physics-Informed GAN (PIGAN) for generating Neuropixels-like signals to support data augmentation.

### Key Features
- Physics-informed regularization: encourages band-limited, smooth, and sparse spike-like waveforms.
- Coordinate-based generator: produces signals as a function of time coordinates and latent codes.
- Stable adversarial training using WGAN-GP with optional spectral and temporal constraints.
- Simple YAML config, reproducible training, and clean modular code.

### Repository Structure
- `neuro_pigan/`: Python package
  - `models/`: Generator and Discriminator
  - `data/`: Neuropixels dataset loader and synthetic fallback
  - `losses/`: GAN loss and physics-informed penalties
  - `training/`: Trainer and physics constraints composition
  - `utils/`: Config, logging, metrics helpers
- `configs/`: Default training configuration(s)
- `scripts/`: Convenience scripts to launch training
- `data/`, `logs/`, `outputs/`: Input and generated artifacts
- `train.py`: Entry point for training

### Quickstart
1) Create a virtual environment and install dependencies
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

2) (Optional) Prepare Neuropixels data
- Place arrays in `data/` as `.npy` or `.pt` with shape `[num_traces, trace_length]`.
- If `data/` is empty, a synthetic dataset will be used for sanity checks.

3) Train with default config
```bash
python train.py --config configs/default.yaml
```

### Configuration
Edit `configs/default.yaml` to adjust:
- `data`: path, sequence length, batch size
- `model`: latent size, hidden sizes, activation
- `gan`: training steps, learning rates, WGAN-GP coefficients
- `physics`: weights for band-limit, smoothness, amplitude range, sparsity
- `device`: `cuda` or `cpu`

### Dataset Format
- Expected file layout:
  - `data/*.npy` or `data/*.pt` where each file is a 2D array `[N, T]`
- The loader will concatenate across files and sample random windows of length `seq_len`.

### Outputs
- Checkpoints and samples are written to `outputs/`
- Logs (loss curves, scalars) are recorded to `logs/`

### Notes on Physics-Informed Losses
This template includes practical, domain-inspired constraints for extracellular potentials:
- Band-limit penalty (keeps most energy within a target frequency band, e.g. 300–6000 Hz)
- Temporal smoothness via second derivative penalties
- Amplitude range penalty (discourages extreme voltages)
- Optional sparsity prior (encourages spiking transients)

You can implement domain-specific constraints in `neuro_pigan/training/physics_constraints.py`.

### License
MIT License.
