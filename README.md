# AttnPINN-for-RUL-Estimation
This repository includes the code and data for the paper "_**A Framework for Remaining Useful Life Prediction Based on Self-Attention and Physics-Informed Neural Networks**_"

![SourceForge Platform](https://img.shields.io/sourceforge/platform/python?color=python&label=python&logo=python)

![GitHub watchers](https://img.shields.io/github/watchers/XinyuanLiao/AttnPINN-for-RUL-Estimation?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/XinyuanLiao/AttnPINN-for-RUL-Estimation?style=social)
![GitHub all releases](https://img.shields.io/github/downloads/XinyuanLiao/AttnPINN-for-RUL-Estimation/total?style=social)
## Abstract
_Prognostics and health management (PHM), as an important technique that can timely make maintenance plans for important equipment and reduce maintenance costs, has attracted more and more attention. Remaining useful life (RUL) prediction as the key of PHM has also been more and more researched. The current mainstream RUL prediction method is data-driven. However, the large number of model parameters, low prediction accuracy, and lack of interpretability of prediction results are common problems of current data-driven methods. Physics-Informed Neural Networks (PINNs) and Self-Attention mechanism, an algorithm that can effectively learn the interactions and differences between features, are introduced in this paper for RUL prediction, achieving fewer parameters, higher prediction accuracy, and better interpretation of prediction results. The RUL prediction framework based on the Self-Attention mechanism and PINNs called AttnPINN proposed in this paper has verified its superiority on the Commercial Modular AeroPropulsion System Simulation (C-MAPSS) dataset._

## Requirements
* matplotlib==3.3.2
* numpy==1.21.6
* scikit_learn==1.0.2
* torch==1.11.0
* torchsummary==1.5.1
## Install Environments
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
In `main.py`, it includes training, predicting and drawing functions.

:sunglasses: By default, only predicting function will run.

## Output

By default, the output will be:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
MultiheadAttention-1  [[-1, 1, 14], [-1, 2, 2]]               0
         LayerNorm-2                [-1, 1, 14]              28
            Linear-3                [-1, 1, 14]             210
              ReLU-4                [-1, 1, 14]               0
            Linear-5                 [-1, 1, 3]              45
================================================================
Total params: 283
Trainable params: 283
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Linear-1                [-1, 1, 10]              50
              Tanh-2                [-1, 1, 10]               0
            Linear-3                [-1, 1, 10]             110
              Tanh-4                [-1, 1, 10]               0
            Linear-5                [-1, 1, 10]             110
              Tanh-6                [-1, 1, 10]               0
            Linear-7                [-1, 1, 10]             110
              Tanh-8                [-1, 1, 10]               0
            Linear-9                [-1, 1, 10]             110
             Tanh-10                [-1, 1, 10]               0
           Linear-11                 [-1, 1, 6]              66
================================================================
Total params: 556
Trainable params: 556
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
MultiheadAttention-1  [[-1, 1, 10], [-1, 2, 2]]               0
         LayerNorm-2                [-1, 1, 10]              20
            Linear-3                [-1, 1, 10]             110
              ReLU-4                [-1, 1, 10]               0
            Linear-5                 [-1, 1, 1]              11
================================================================
Total params: 141
Trainable params: 141
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.00
Estimated Total Size (MB): 0.00
----------------------------------------------------------------
Test_RMSE: 18.37,   Score: 2058.5
```

If you want to train the model by yourself:hammer::hammer:, you can uncomment the train function in _**[main.py](https://github.com/XinyuanLiao/AttnPINN-for-RUL-Estimation/blob/main/main.py)**_.

```
#pinn.train(1000) =>pinn.train(1000)
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
|Method|RMSE|Score|Parameters|
|-|-|-|-|
|DCNN[(Li et al., 2018)](https://www.sciencedirect.com/science/article/pii/S0951832017307779)|23.31|12466|72.7K|
RNN-Autoencoder[(Yu et al.. 2020)](https://www.sciencedirect.com/science/article/pii/S0951832019307902)|22.15|2901|378.0K
GCU-Transformer[(Mo et al.,2021)](https://link.springer.com/article/10.1007/s10845-021-01750-x)|24.86|N/A|399.7K
MCLSTM[(Sheng et al., 2021)](https://www.sciencedirect.com/science/article/pii/S0951832021004439)|23.81|4826|N/A
Double attention-Transformer[(Liu et al., 2022)](https://www.sciencedirect.com/science/article/pii/S0951832022000102)|19.86|1741|N/A
e-RULENet[(Natsumeda, 2022)](https://ieeexplore.ieee.org/abstract/document/9905797/)|20.80|**1554**|32.3K
PDE-PHM[(Cofre-Martel et al., 2021)](https://www.hindawi.com/journals/sv/2021/9937846/)|25.58|N/A|1,066
AttnPINN[(proposed framwork)]()|**18.58**|2019|**986**
## Citation
