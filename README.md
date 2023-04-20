# AttnPINN-for-RUL-Estimation
This repository includes the code and data for the paper "_**A Framework for Remaining Useful Life Prediction Based on Self-Attention and Physics-Informed Neural Networks**_"

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />
## Abstract
Prognostics and health management (PHM), as an important technique that can timely make maintenance plans for important equipment and reduce maintenance costs, has attracted more and more attention. Remaining useful life (RUL) prediction as the key of PHM has also been more and more researched. The current mainstream RUL prediction method is data-driven. However, the large number of model parameters, low prediction accuracy, and lack of interpretability of prediction results are common problems of current data-driven methods. Due to the outstanding advantages of the Self-Attention mechanism in reducing the amount of parameters and learning the potential distribution of data and the good effect of the Physics-Informed Neural Networks (PINNs) in improving the prediction accuracy of the network, this paper introduces the Self-Attention mechanism and PINNs in RUL prediction to achieve fewer parameters, higher prediction accuracy and better interpretation of prediction results. The RUL prediction framework based on the Self-Attention mechanism and PINNs called AttnPINN proposed in this paper has verified its superiority on the Commercial Modular AeroPropulsion System Simulation (C-MAPSS) dataset.

## Requirements
* matplotlib==3.3.2
* numpy==1.21.6
* scikit_learn==1.0.2
* torch==1.11.0
* torchsummary==1.5.1
## Install environments
If you want to install the required environments one by one:thinking:, you can copy the following codes:
```
pip install matplotlib==3.3.2
pip install numpy==1.21.6
pip install scikit_learn==1.0.2
pip install torch==1.11.0
pip install torchsummary==1.5.1
```
or use this:stuck_out_tongue::
```
pip install -r requirements.txt
```
## Test
Running the project with the following code:
```
python main.py
```
In [main.py](https://github.com/XinyuanLiao/AttnPINN-for-RUL-Estimation/blob/main/main.py), it includes training, predicting and drawing functions.

:sunglasses: By default, only predicting function will run.

## Comparisons with state-of-the-art methods
|Method|RMSE|Score|Parameters|
|-|-|-|-|
|DCNN[(Li et al., 2018)](https://www.sciencedirect.com/science/article/pii/S0951832017307779)|23.31|12466|72.7K|
RNN-Autoencoder[(Yu et al.. 2020)](https://www.sciencedirect.com/science/article/pii/S0951832019307902)|22.15|2901|378.0K
GCU-Transformer[(Mo et al.,2021)](https://link.springer.com/article/10.1007/s10845-021-01750-x)|24.86|N/A|399.7K
MCLSTM[(Sheng et al., 2021)](https://www.sciencedirect.com/science/article/pii/S0951832021004439)|23.81|4826|N/A
Double attention-Transformer[(Liu et al., 2022)](https://www.sciencedirect.com/science/article/pii/S0951832022000102)|19.86|1741|N/A
e-RULENet[(Natsumeda, 2022)](https://ieeexplore.ieee.org/abstract/document/9905797/)|20.80|**1554**|32.3K
PDE-PHM[(Cofre-Martel et al., 2021)](https://www.hindawi.com/journals/sv/2021/9937846/)|25.58|N/A|1,066
AttnPINN[(proposed framwork)]()|**18.58**|2019|**1,030**
## Citation
