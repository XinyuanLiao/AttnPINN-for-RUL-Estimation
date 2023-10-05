![SourceForge Platform](https://img.shields.io/sourceforge/platform/python?color=python&label=python&logo=python)
![](https://img.shields.io/hexpm/l/plug)
# AttnPINN for RUL Estimation
This repository includes the code and data for the paper "_**[Remaining useful life with self-attention assisted physics-informed neural network](https://www.sciencedirect.com/science/article/pii/S1474034623003233?dgcid=author)**_"

## Dataset
This paper just evaluated the FD004 sub-dataset and the whole dataset can be found [**here**](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

## Abstract
_Remaining useful life (RUL) prediction as the key technique of prognostics and health management (PHM) has been extensively investigated. The application of data-driven methods in RUL prediction has advanced greatly in recent years. However, a large number of model parameters, low prediction accuracy, and lack of interpretability of prediction results are common problems of current data-driven methods. In this paper, we propose a Physics-Informed Neural Networks (PINNs) with Self-Attention mechanism-based hybrid framework for aircraft engine RUL prognostics. Specifically, the self-attention mechanism is employed to learn the differences and interactions between features, and reasonably map high-dimensional features to low-dimensional spaces. Subsequently, PINN is utilized to regularize the end-to-end prediction network, which maps features to RUL. The RUL prediction framework termed AttnPINN has verified its superiority on the Commercial Modular AeroPropulsion System Simulation (C-MAPSS) dataset. It achieves state-of-the-art prediction performance with a small number of parameters, resulting in computation-light features. Furthermore, its prediction results are highly interpretable and can accurately predict failure modes, thereby enabling precise predictive maintenance._

## Configuration
* matplotlib==3.3.2
* numpy==1.21.6
* scikit_learn==1.0.2
* torch==1.11.0
* torchsummary==1.5.1

If you want to install the required environments one by one, you can copy the following codes:
```
pip install matplotlib==3.3.2
pip install numpy==1.21.6
pip install scikit_learn==1.0.2
pip install torch==1.11.0
pip install torchsummary==1.5.1
```
or use this:
```
pip install -r requirements.txt
```
## Quick Start
Running the project with the following code:
```
python main.py
```
In `main.py`, it includes training, predicting and drawing functions.

 By default, only predicting function will run and the output will be:

```
Test_RMSE: 18.37,   Score: 2058.5
```

If you want to train the model by yourself:hammer::hammer:, you can uncomment the train function in _**[main.py](https://github.com/XinyuanLiao/AttnPINN-for-RUL-Estimation/blob/main/main.py)**_.

```
#pinn.train(1000) => pinn.train(1000)
```

And then, the output will be:

```
It: 0,   Valid_RUL_RMSE: 100.92
It: 1,   Valid_RUL_RMSE: 99.87
It: 2,   Valid_RUL_RMSE: 40.89
It: 3,   Valid_RUL_RMSE: 40.76
It: 4,   Valid_RUL_RMSE: 40.74
It: 5,   Valid_RUL_RMSE: 40.74
It: 6,   Valid_RUL_RMSE: 35.48
It: 7,   Valid_RUL_RMSE: 20.47
It: 8,   Valid_RUL_RMSE: 18.70
It: 9,   Valid_RUL_RMSE: 18.27
···
```

## Comparisons with State-of-the-art Methods
|Method|RMSE|Score|Parameters|FLOPs|
|-|-|-|-|-|
|DCNN[(Li et al., 2018)](https://www.sciencedirect.com/science/article/pii/S0951832017307779)|23.31|12466|35.05k|174.76k
RNN-Autoencoder[(Yu et al.. 2020)](https://www.sciencedirect.com/science/article/pii/S0951832019307902)|22.15|2901|378.0K|N/A
GCU-Transformer[(Mo et al.,2021)](https://link.springer.com/article/10.1007/s10845-021-01750-x)|24.86|N/A|399.7K|393.39
Double attention-Transformer[(Liu et al., 2022)](https://www.sciencedirect.com/science/article/pii/S0951832022000102)|19.86|1741|N/A|N/A
e-RULENet[(Natsumeda, 2022)](https://ieeexplore.ieee.org/abstract/document/9905797/)|20.80|1554|32.3K|N/A
CNN-BiLSTM-3DAttention[(You et al., 2023)](https://ieeexplore.ieee.org/abstract/document/10190349)|20.24|1710|151.9k|170.3k
MTSTAN[(Li et al., 2023)](https://www.sciencedirect.com/science/article/pii/S1474034623000265)|18.85|**1446**|N/A|N/A
PINN-PHM[(Cofre-Martel et al., 2021)](https://www.hindawi.com/journals/sv/2021/9937846/)|25.58|N/A|**1,066**|N/A
AttnPINN[(proposed framework)](https://www.sciencedirect.com/science/article/pii/S1474034623003233?dgcid=author)|**18.37**|2059|2,260|**1728**

## Cite Repository
Please, cite this repository using:
```
@article{LIAO2023102195,
title = {Remaining useful life with self-attention assisted physics-informed neural network},
journal = {Advanced Engineering Informatics},
volume = {58},
pages = {102195},
year = {2023},
issn = {1474-0346},
doi = {https://doi.org/10.1016/j.aei.2023.102195},
url = {https://www.sciencedirect.com/science/article/pii/S1474034623003233},
author = {Xinyuan Liao and Shaowei Chen and Pengfei Wen and Shuai Zhao}
}
```

