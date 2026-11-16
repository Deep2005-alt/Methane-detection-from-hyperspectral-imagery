## Methane-detection-from-hyperspectral-imagery
H-MRCNN introduces fast algorithms to analyze large-area hyper-spectral information and methods to autonomously represent and detect CH4 plumes. This repo contains 2 methods for processing different type of data, Single detector works on 4-channels data and Ensemble detectors works on 432-channels raw hyperspectral data recorded from AVIRIS-NG instrument. 
### [**Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery**](https://openaccess.thecvf.com/content_WACV_2020/papers/Kumar_Deep_Remote_Sensing_Methods_for_Methane_Detection_in_Overhead_Hyperspectral_WACV_2020_paper.pdf)
[Satish Kumar*](https://www.linkedin.com/in/satish-kumar-81912540/), [Carlos Torres*](https://torrescarlos.com), [Oytun Ulutan](https://sites.google.com/view/oytun-ulutan), [Alana Ayasse](https://www.linkedin.com/in/alana-ayasse-18370972/), [Dar Roberts](http://geog.ucsb.edu/archive/people/faculty_members/roberts_dar.htm), [B S Manjunath](https://vision.ece.ucsb.edu/people/bs-manjunath).

Official repository of our [**WACV 2020**](https://openaccess.thecvf.com/content_WACV_2020/papers/Kumar_Deep_Remote_Sensing_Methods_for_Methane_Detection_in_Overhead_Hyperspectral_WACV_2020_paper.pdf) paper.

<img src="overview.gif" width="700">

This repository includes:
* Source code of single-detector and ensemble detectors(H-MRCNN) built on Mask-RCNN.
* Training code for single-detector and ensemble detectors(H-MRCNN)
* Pre-trained ms-coco weights of Mask-RCNN
* Annotation generator to read-convert mask annotation into json.
* Modified spectral library of python
* Example of training on your own dataset

![supported versions](https://img.shields.io/badge/python-(3.5--3.8)-brightgreen/?style=flat&logo=python&color=green)
![Library](https://img.shields.io/badge/Library-TensorFlow-blue)
![GitHub license](https://img.shields.io/cocoapods/l/AFNetworking)

The whole repo folder structure follows the same style as written in the paper for easy reproducibility and easy to extend. If you use it in your research, please consider citing our paper (bibtex below)

## Project Details

### 🛠 Technical Stack
* **Language:** Python (3.5 – 3.8)
* **Deep Learning Framework:** TensorFlow $\le 1.8$
* **Hardware Acceleration:** CUDA 9.0 & cuDNN
* **Core Architecture:** Mask R-CNN
* **Specialized Libraries:** Modified `spectral` library for hyperspectral data cubes.

### 🌟 Key Features & Methodology
The project provides two distinct detection pipelines:
1. **Single-Detector Pipeline:** Works on 4-channel data.
2. **Ensemble-Detector Pipeline (H-MRCNN):** 
   * Works on raw 432-channel hyperspectral data.
   * Processes data in overlapping sets (e.g., 50 channels per set with 25-channel overlap).
   * **Pipeline:** Matched Filter & Ortho-correction $\rightarrow$ 3-Channel Tile Generation (Terrain, Local Norm, Global Norm) $\rightarrow$ Mask R-CNN Detection.

### 📂 Data & Tile Catalog
| Category | Path | Description |
| :--- | :--- | :--- |
| **Raw Data** | `ensemble_detectors/data/raw_data/` | Original 432-channel hyperspectral files. |
| **Ground Truth** | `ensemble_detectors/data/gt_data/` | Mask annotations used for training. |
| **Spectral Lib** | `ensemble_detectors/data/gas_signature/` | Contains `methane_signature.txt`. |
| **Processed Tiles**| `ensemble_detectors/data/train_data/` | 3-channel training tiles. |
| **Annotations** | `ensemble_detectors/src/gt_jsonfile/` | JSON files for Mask R-CNN. |
| **Model Weights** | `ensemble_detectors/src/Algorithm_3_mrcnn/logs/` | Weights for each band-set detector. |

## Citing
If this work is useful to you, please consider citing our paper:
```
@inproceedings{kumar2020deep,
  title={Deep Remote Sensing Methods for Methane Detection in Overhead Hyperspectral Imagery},
  author={Kumar, Satish and Torres, Carlos and Ulutan, Oytun and Ayasse, Alana and Roberts, Dar and Manjunath, BS},
  booktitle={2020 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  pages={1765--1774},
  year={2020},
  organization={IEEE}
}
```

### Requirements
- Linux or macOS with Python ≥ 3.6
- Tensorflow <= 1.8
- CUDA 9.0
- cudNN (compatible to CUDA)

### Installation
1. Clone this repository
2. Install dependencies
```
pip install -r requirements.txt
```
#### Single-detector
Running single-detector is quite simple. Follow the [README.md](https://github.com/satish1901/Methane-detection-from-hyperspectral-imagery/blob/master/single_detector/README.md) in single_detector folder
```
single_detector/README.md
```

#### Ensemble-detector
For Running ensemble-detector we need some pre-processing. Follow the [README.md](https://github.com/satish1901/Methane-detection-from-hyperspectral-imagery/blob/master/ensemble_detectors/README.md) in emsemble_detector folder
```
ensemble_detector/README.md
```
