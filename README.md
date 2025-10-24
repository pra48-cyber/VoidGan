# Inpainting


[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
![Flask](https://img.shields.io/badge/Flask-2.3+-red)


Hybrid Transformer-CNN for Digital Elevation Model (DEM) inpainting & reconstruction — a PyTorch implementation that:
- Automatically computes dataset normalization statistics
- Trains a hybrid Transformer-UNet architecture
- Corrupts input DEM tiles with a learned mask (void regions) and reconstructs them with a diffusion model
- Saves only the single *best* model (lowest average epoch loss)
- Produces visual progress snapshots every epoch

🎯 Purpose: Repair and generate missing DEM regions using a diffusion-based Hybrid Transformer-UNet.

---

## Highlights
- ✨ Clean, modular PyTorch code with training loop, dataset loader, and utilities.
- 🧮 Automatic calculation of normalization mean & std from GeoTIFF tiles.
- 🏆 Keeps only the best model based on average training loss per epoch.
- 🖼️ Visual outputs per epoch for easy inspection of progress.

---

## Table of contents
- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Repository layout (main parts)](#repository-layout-main-parts)
- [Configuration](#configuration)
- [How it works (high-level)](#how-it-works-high-level)
- [Model architecture summary](#model-architecture-summary)
- [Training command & usage](#training-command--usage)
- [Tips & troubleshooting](#tips--troubleshooting)
- [Acknowledgements & license](#acknowledgements--license)

---

## Quick start

1. Install dependencies:
```bash
!pip install torch torchvision rasterio matplotlib tqdm
# Optionally: pip install accelerate  # if you use distributed training helpers
```

2. Place your DEM GeoTIFF tiles (.tif) in a folder and update the DATA_PATH in the config.

3. Run the training script (example):
```.
You can use directly that notebook for training and testing

```

Outputs:
- Normalization stats → saved to `STATS_PATH` (JSON)
- Best model weights → saved to `MODEL_SAVE_PATH` (single .pth)
- Visual epoch snapshots → saved into `VISUAL_PATH`

---

## Requirements
- Python 3.8+
- PyTorch (1.12+ recommended for CUDA/mps support)
- rasterio (for reading GeoTIFF)
- numpy, matplotlib, tqdm
- GPU recommended for reasonable training speed

Install:
```bash
pip install torch torchvision rasterio numpy matplotlib tqdm
```

---

## Repository layout (main parts)
- train_diff_dem.py (main script — the code you provided)
  - Model: `HybridTransformerUNet` — UNet + Transformer bottleneck
  - Diffusion process: `Diffusion` (linear beta schedule)
  - Dataset: `DEMDataset` reads .tif files and resizes/caches them to training size
  - Utilities:
    - `calculate_norm_stats()` — computes mean & std and writes JSON
    - `generate_void_mask()` — random rectangular voids for training
    - `generate_visuals()` — assembles and saves progress images
  - Training logic saves the best model using `avg_epoch_loss`

---

## Configuration
All training-related parameters are in the `TrainingConfig` class:
```python
class TrainingConfig:
    DATA_PATH = "/path/to/your/tif/folder"
    OUTPUT_PATH = "/path/to/output"
    VISUAL_PATH = "/path/to/output/visuals"
    STATS_PATH = "/path/to/output/norm_stats.json"
    MODEL_SAVE_PATH = "/path/to/output/best_model.pth"
    EPOCHS = 300
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    IMG_SIZE = 128
    TIMESTEPS = 1000
    MIN_VOID_RATIO = 0.25
    MAX_VOID_RATIO = 0.60
    VIS_TIMESTEPS = 250
```

Customize these values to your hardware and dataset size. Important ones:
- IMG_SIZE: patch size used for training (resized with bilinear)
- TIMESTEPS: diffusion timesteps (1000 is standard; lower for faster experiments)
- VIS_TIMESTEPS: timesteps used when generating visuals for display
- MIN_VOID_RATIO / MAX_VOID_RATIO: control how much of the tile is masked during training

---

## How it works (high-level)
1. The script computes mean and std over all `.tif` tiles and stores them in a JSON file.
2. DEM tiles are loaded, resized to IMG_SIZE, normalized, and randomly augmented (flips).
3. During training:
   - A rectangular void mask is generated for each sample.
   - The void region is replaced by diffusion noise (q_sample).
   - The model learns to predict the added noise (denoising objective / MSE) but the loss is computed only over masked pixels.
4. The best-performing model per epoch (measured via average training loss) is saved as the canonical checkpoint.
5. Visual snapshots are generated every epoch (ground truth, corrupted input, generated output).

---

## Model architecture summary
- Input channels: 2 (concatenation of corrupted image and its conditional input)
- Time embedding: sinusoidal positional embeddings → MLP
- Encoder: 4 DoubleConv blocks with GroupNorm + GELU
- Bottleneck: DoubleConv → flattened → TransformerEncoder (4 layers, 4 heads, d_model=1024)
- Decoder: ConvTranspose upsampling + DoubleConv with skip connections
- Output: 1-channel reconstruction map (same spatial size as input)
- Loss: MSE between predicted noise and actual noise, applied only on masked pixels

Why hybrid? The CNNs capture local texture/edge structure while the transformer models capture more global context inside the bottleneck patch representation.

---

## Training & usage examples

Run training:
```bash
python train_diff_dem.py
```

If you want to run within a notebook cell:
```python
# in a Jupyter notebook cell
%run train_diff_dem.py
```

Evaluate or generate samples after training:
- Load the best model:
```python
model = HybridTransformerUNet(img_channels=2, time_emb_dim=32)
model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
model.to(device).eval()
```
- Use `generate_visuals()` or write a custom inference loop for larger tiles.

---

## Dataset expectations
- The script expects many single-band GeoTIFFs (`*.tif`) inside `DATA_PATH`.
- Each file is read as a single-band DEM; any reprojection / alignment should be done upstream.
- A simple example layout:
```
/data/dem_tiles/
  tile_000.tif
  tile_001.tif
  ...
```

---

## Tips & troubleshooting
- Out of memory? Reduce BATCH_SIZE or IMG_SIZE, or use gradient accumulation.
- Slow normalization stats? The script reads images in batches; use faster I/O or smaller batch_size in calculate_norm_stats if needed.
- If rasterio fails to read files: check the GeoTIFF format and GDAL backend.
- For scientific reproducibility: set seeds for torch, numpy, and random:
```python
seed = 42
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
```
- If you want model checkpoints every N epochs in addition to the best model — add a periodic save in the training loop.

---

## Recommended experiments
- Reduce TIMESTEPS to speed up prototyping (e.g., 100 or 250) — then scale up to 1000 for final runs.
- Try different mask shapes (irregular, circular, mixed sizes).
- Replace GroupNorm(1) with GroupNorm(num_groups) or InstanceNorm for different behavior.
- Increase transformer depth or heads to capture larger context (if GPU permits).

---

## Contributing
- Bug fixes, feature requests, and pull requests are welcome!
- Please add tests for new functions and keep changes modular.
- If you change default config paths, update README and example commands.

---

## Acknowledgements
- Diffusion modeling concepts inspired by modern denoising diffusion literature.
- Transformer + UNet hybridization inspired by recent segmentation & restoration works.

---
