# BD-Net


This is the repository for our paper entitled [Nonlinear Regression of Remaining Surgical Duration via Bayesian LSTM-Based Deep Negative Correlation Learning](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_40), which was already acceped in MICCAI 2022.


![avatar](https://github.com/jywu511/BD-Net/blob/main/BDNet.png)



In this paper, we address the problem of estimating remaining surgical duration (RSD) from surgical video frames. We propose a Bayesian long short-term memory (LSTM) network-based Deep Negative Correlation Learning approach called BD-Net for accurate regression of RSD prediction as well as estimating prediction uncertainty. Our method aims to extract discriminative visual features from surgical video frames and model the temporal dependencies among frames to improve the RSD prediction accuracy. To this end, we propose to ensemble a group of Bayesian LSTMs on top of a backbone network by the way of deep negative correlation learning (DNCL). More specifically, we deeply learn a pool of decorrelated Bayesian regressors with sound generalization capabilities through managing their intrinsic diversities. BD-Net is simple and efficient. After training, it can produce both RSD prediction and uncertainty estimation in a single inference run.

## Inference

You can download our trained model on [One Drive](https://sjtueducn-my.sharepoint.com/:f:/g/personal/sjtuwjy_sjtu_edu_cn/Ev4feqx0tetDrvc_LxXiGZYBLLGrqB4FUpxHRZXDSOFApA?e=BDzVqz)
