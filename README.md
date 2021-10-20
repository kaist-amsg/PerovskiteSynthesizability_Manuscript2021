PerovskiteSynthesizability_Manuscript2021
=========================================
In hope to quantify the synthesizablility of virtual crystals, we implemented transfer learning and positive-unlabeled (PU) learning to predict synthesizability. This repository contains code to reproduce the result in our manuscript.

Developers
----------
Geun Ho Gu (ggu@udel.edu) <- Current maintainer

Dependencies
------------
-  Python3
-  Scikit-learn
-  Numpy
-  Pytorch
-  Pymatgen
-  Pytorch_scatter (https://github.com/rusty1s/pytorch_scatter)

Installation
------------
1. Clone this repository:
```
git clone https://github.com/kaist-amsg/PerovskiteSynthesizability_Manuscript2021
```
Data Availablility
------------------
In the ./data folder, we have all raw crystal data in cif format. The source of the cif data (e.g. MP, OQMD, AFLOW ids) are in the cif_sources.json. The provided python codes contain preprocessing, model training, and synthesizability prediction.

Guide
-----
1PU_Split.py splits the data (splits_Perov_All.json) in accordance to the inductive PU learning method. 2Train.py trains 100 PU learning model. The model weights trained with MP data is in base_weights, which are then transferred and retrained, weight of which are saved in weights folder. 3Predict.py makes prediction to all the data, which is summarized in the file "prediction.csv."

- ./data folder contains crystal data (*.cif), elemental combination of each crystal file (Perov_All_ABC.json), and the source of each cif file (cif_sources.json), and the icsd label (id_prop.csv).
- ./gcnn folder contains the python module for data preprocessing and model construction.
- ./predict folder contains model prediction in json file for all the crystal data.
- ./tests folder contains prediction result for the held-out test sets.
- ./base_weights contains model weights trained with MP data, and "./weights" contains retrained paramaters after trasnferring.
The CL score prediction result is in the SI of the manuscript, but it is also tabulated in the "prediction.csv."
Publications
------------
If you use this code, please cite:

Geun Ho Gu, Jidon Jang, Juhwan Noh, Aron Walsh, and Yousung Jung. Perovskite Synthesizability using Graph Neural Networks, In preparation

