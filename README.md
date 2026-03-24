# ESPD-DETR: Enhancing Small-Object Representation...

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXX)

Official implementation of the paper submitted to **The Visual Computer**.

## 🚀 Highlights
- **ERF-Block**: Efficient feature extraction for UAV images.
- **CGFM**: Context-guided multi-scale fusion.
- **SOEP**: SPDConv-based pyramid for small objects.

## 🛠️ Setup
1. `pip install -r requirements.txt`
2. Prepare VisDrone dataset in `data/`
3. Run `python train.py --config configs/espd_detr_l.yaml`