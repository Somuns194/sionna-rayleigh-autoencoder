# Rayleigh Fading Autoencoder Communication System

End-to-end autoencoder over Rayleigh block-fading channels with perfect vs. imperfect CSI. Compares BCE backpropagation and RL policy gradient training. Built with NVIDIA Sionna.

## Overview

Conventional communication systems optimize each module independently. End-to-end autoencoders jointly optimize the transmitter and receiver, which can outperform modular designs. However, most public implementations cover only AWGN or differentiable channels.

This project extends Sionna's `Autoencoder.ipynb` to Rayleigh block-fading and introduces a **model mismatch** experiment:

- **BCE mismatch**: trained with perfect CSI, tested with imperfect CSI.
- **RL robust**: trained directly with imperfect CSI, tested with the same.

The comparison shows how each training method handles channel estimation errors.

## Key Results

| Scenario | Finding |
|----------|---------|
| **AWGN** | Autoencoders (BCE/RL) outperform conventional LDPC+64QAM by ~1 dB at BLER=1e‑3. |
| **Rayleigh + Perfect CSI** | All three systems perform similarly. Deep fading masks constellation shaping gains. |
| **Rayleigh + Imperfect CSI** | RL reduces BLER by ~35% compared to BCE model mismatch at 24.5 dB (0.367 vs 0.565). RL has no error floor; BCE does. |
| **Constellations** | RL under imperfect CSI learns a more dispersed, Gaussian‑like constellation, indicating robustness to CSI errors. |

Full analysis is in the [project report](./瑞利衰落自编码器项目报告.md) (Chinese).

## Requirements

- Python 3.13
- PyTorch (CUDA recommended)
- Sionna 2.0+
- matplotlib, numpy, pickle

On Windows, `torch.compile` is disabled in the code for compatibility.

## Quick Start

```bash
git clone https://github.com/yourname/rayleigh-autoencoder.git
cd rayleigh-autoencoder

# Install dependencies (virtual environment recommended)
pip install torch sionna matplotlib numpy

# Smoke test to verify the environment
python rayleigh_autoencoder.py smoke

# Train all 5 models (~8 minutes on RTX 4060)
python rayleigh_autoencoder.py train

# Evaluate and generate plots
python rayleigh_autoencoder.py eval

# Run the full pipeline: smoke → train → eval
python rayleigh_autoencoder.py all
Trained weights are saved in weights/. Results are stored in rayleigh_results.pkl; figures are saved as rayleigh_results.png and rayleigh_constellations.png.

## Project Structure
.
├── rayleigh_autoencoder.py # Main script (train/eval/plot)
├── weights/ # 5 trained model weights (.pt)
├── rayleigh_results.pkl # All BLER simulation data
├── rayleigh_results.png # BLER curves for 3 scenarios
├── rayleigh_constellations.png # Learned constellations
├── training_log.txt # Training logs
├── eval_log.txt # Evaluation logs per SNR
├── 瑞利衰落自编码器项目报告.md # Full technical report (Chinese)
├── .gitignore # Ignore cache and temporary files
└── LICENSE # Open-source license
