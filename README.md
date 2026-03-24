# ESPD-DETR: Enhancing Small-Object Representation with Context-Guided Feature Pyramids for UAV Imagery

[![DOI](https://zenodo.org/badge/1190321684.svg)](https://doi.org/10.5281/zenodo.19201263)

Official implementation of the paper submitted to **The Visual Computer**.

## 🚀 Highlights
- **ERF-Block**: Efficient feature extraction for UAV images.
- **CGFM**: Context-guided multi-scale fusion.
- **SOEP**: SPDConv-based pyramid for small objects.

## 🛠️ Setup
1. `pip install -r requirements.txt`
2. Prepare VisDrone dataset in `data/`
3. Run `python train.py --config configs/espd_detr_l.yaml`
