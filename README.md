[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Transfer Reinforcement Learning for MOMDPs with Time-varying Interval-valued Parameters and Its Application in Pandemic Control

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported on in the paper 
[Transfer Reinforcement Learning for MOMDPs with Time-varying Interval-valued Parameters and Its Application in Pandemic Control](https://doi.org/10.1287/ijoc.2022.0236) by Mu Du, Hongtao Yu, Nan Kong. 


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

https://doi.org/10.1287/ijoc.2022.0236

https://doi.org/10.1287/ijoc.2022.0236.cd

Below is the BibTex for citing this snapshot of the respoitory.

```
@article{TRL for MOMDP-TVIVP,
  author =        {Mu Du, Hongtao Yu, Nan Kong},
  publisher =     {INFORMS Journal on Computing},
  title =         {{Transfer Reinforcement Learning for MOMDPs with Time-varying Interval-valued Parameters and Its Application in Pandemic Control}},
  year =          {2024},
  doi =           {10.1287/ijoc.2022.0236.cd},
  url =           {https://github.com/INFORMSJoC/2022.0236},
}  
```

## Description

This repository includes the source code and computational results of transfer reinforcement learning approach for randomly generated numerical instances presented in the paper. The codes and random instance genereation work in a Miniforge3 environment with Python 3.9 in a M1 mac platform.

## Numerical Results on Solution Optimality and Robustness

The _**exp1_case**_ folder contains the data, script and codes for numerical results in Section 4.2 & 4.4.
1. The Jupyter Notebook files [DRL_uncertainty_result](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/DRL_uncertainty_result.ipynb) and [TRL_uncertainty_result](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/TRL_uncertainty_result.ipynb) are for testing the performance of the TRL under different uncertainty intervals (**Figure 4.1 and Figure 4.4**). The training and testing process of network are recored in these files. Please excute the codes in former file first for network offline training and then excute the codes in later file for network performance testing. 
2. The Jupyter Notebook files [DRL_out_of_sample](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/DRL_out_of_sample.ipynb) and [TRL_out_of_sample](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/TRL_out_of_sample.ipynb) are for testing the performance of the TRL under different out-of-sample scenarios with bias to intervals (**Figure 4.2**). The training and testing process of network are recored in these files. Please excute the codes in former file first for network offline training and then excute the codes in later file for network performance testing. 
3. The Jupyter Notebook files [DRL_out_of_sample](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/DRL_out_of_sample.ipynb) and [TRL_out_of_sample_result distribution shift](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/distribution%20shift/TRL_out_of_sample_result%20distribution%20shift.ipynb) are for testing the performance of the TRL under different out-of-sample scenarios with distributional shifts (**Figure 4.3**). The training and testing process of network are recored in these files. Please excute the codes in former file first for network offline training and then excute the codes in later file for network performance testing. 

## Numerical Results on Algorithm efficiency and Scability

The _**exp2_case**_ folder contains the data, script and codes for numerical results of _**scaling up instances**_ in Section 4.3 & 4.4.

| 24*20 dimensions | 48*40 dimensions | 72*60 dimensions | 96*80 dimensions | 120*100 dimensions | 144*120 dimensions |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [Offline_stage_4patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_4patch)  | [Offline_stage_8patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_8patch.ipynb)  | [Offline_stage_12patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_12patch.ipynb) | [Offline_stage_16patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_16patch.ipynb) |[Offline_stage_20patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_20patch.ipynb) | [Offline_stage_24patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_24patch.ipynb) |
| [Test_online_4patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_4patch.ipynb)  | [Test_online_8patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_8patch.ipynb) | [Test_online_12patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_12patch.ipynb) | [Test_online_16patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_16patch.ipynb) | [Test_online_20patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_20patch.ipynb) |[Test_online_24patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_24patch.ipynb) |

The above Jupyter Notebook files record the training and testing process of the TRL on scaling up instance with factor n=4, 8, 12, 16, 20, 24 (**Table 4.1 and Figure 4.5**).  Please excute the codes in **"Offline_stage_xxpatch"** first for network offline training and then excute the codes in **"Test_online_xxpatch"** for network performance testing. 

## Application to a Pandemic Control Case

The _**exp3_case**_ folder contains the data, script and codes for numerical results in Section 5.

The following Jupyter Notebook files:

[Offline_stage_high](https://github.com/yuht1993/2022.0236/blob/patch-1/exp3_case/Offline_stage_high.ipynb),  [Test_online_high](https://github.com/yuht1993/2022.0236/blob/patch-1/exp3_case/Test_online_high.ipynb), [Offline_stage_low](https://github.com/yuht1993/2022.0236/blob/patch-1/exp3_case/Offline_stage_low.ipynb), [Test_online_low](https://github.com/yuht1993/2022.0236/blob/patch-1/exp3_case/Test_online_low.ipynb)

record the training and testing process of the TRL under high-level and low-level budget scenario (**Table 5.1**). Please excute the codes in **"Offline_stage_xxx"** first for network offline training and then excute the codes in **"Test_online_xxx"** for network performance testing.

## 

> [!NOTE]
> Due to file upload size limitations, we have stored the initialization and training results of neural network weights on a cloud drive. Please refer to the Readme file in the **network initialization** and **result** folders in exp1_case, exp2_case, and exp3_case for details about these results.
> 
> If there is a need to retrain the neural network, please modify the code "istrain=1" to "istrain=0" in line 12 of the Train_main file in exp1_case, exp2_case, and exp3_case forlers.
> 
> For any issue in replication of these results, please contact [yuht@dlut.edu.cn](yuht@dlut.edu.cn)
