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
3. Run `python train.py --config configs/espd_detr.yaml`

 ## 📦 Pre-trained Models
The pre-trained weights for ESPD-DETR on the VisDrone2019 dataset are available in the [Releases](https://github.com/pytyty/espd_detr/releases) page. You can download `best.pt` and place it in the `weights/` directory for evaluation.

## 🚀 Evaluation
To evaluate the model on the VisDrone validation set, run:
```bash
python val.py --config configs/espd_detr.yaml --weights weights/best.pt
```
## 🔗 Citation
If you find our work, code, or pre-trained models useful for your research, please consider citing our paper submitted to The Visual Computer:
@article{espddetr2026,
  title={ESPD-DETR: Enhancing Small-Object Representation with Context-Guided Feature Pyramids for UAV Imagery},
  author={Yang Tian，Guanxun Cui},
  journal={The Visual Computer},
  year={2026},
  note={Under Review}
}
