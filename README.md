# AttnPINN-for-RUL-estimation
A framework for remaining useful life prediction based on Self-Attention and Physics-Informed Neural Networks
# Abstract
Prognostics and health management (PHM), as an important technique that can timely make maintenance plans for important equipment and reduce maintenance costs, has attracted more and more attention. Remaining useful life (RUL) prediction as the key of PHM has also been more and more researched. The current mainstream RUL prediction method is data-driven. However, the large number of model parameters, low prediction accuracy, and lack of interpretability of prediction results are common problems of current data-driven methods. Due to the outstanding advantages of the Self-Attention mechanism in reducing the amount of parameters and learning the potential connection of data and the good effect of the Physics-Informed Neural Networks (PINNs) in improving the prediction accuracy of the network, this paper introduces the Self-Attention mechanism and PINNs in RUL prediction to achieve fewer parameters, higher prediction accuracy and better interpretation of prediction results. The RUL prediction framework based on the Self-Attention mechanism and PINNs called AttnPINN proposed in this paper has verified its superiority on the Commercial Modular AeroPropulsion System Simulation (C-MAPSS) dataset.
# Framework
![Alt text](图片链接 "The framework proposed in this work")
# Experiment Result
|Method|RMSE|Score|Parameters|
|-|-|-|-|
|DCNN(Li et al., 2018)|23.31|12466|72.7K|
RNN-Autoencoder(Yu et al.. 2020)|22.15|2901|378.0K
GCU-Transformer(Mo et al.,2021)|24.86|N/A|399.7K
MCLSTM(Sheng et al., 2021)|23.81|4826|N/A
Double attention-Transformer(Liu et al., 2022)|19.86|1741|N/A
e-RULENet(Natsumeda, 2022)|20.80|1554|32.3K
PDE-PHM(Cofre-Martel et al., 2021)|25.58|N/A|1,066
AttnPINN(proposed framwork)|18.58|2019|1,030
