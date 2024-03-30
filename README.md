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

The _**exp1_case**_ folder contains the data, script and codes for numercail results in Section 4.2 & 4.4.
1. The Jupyter Notebook files [DRL_uncertainty_result](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/DRL_uncertainty_result.ipynb) and [TRL_uncertainty_result](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/TRL_uncertainty_result.ipynb) are for testing the performance of the TRL under different uncertainty intervals (**Figure 4.1 and Figure 4.4**). The training and testing process of network are recored in these files. Please excute the codes in former file first for network offline training and then excute the codes in later file for network performance testing. 
2. The Jupyter Notebook files [DRL_out_of_sample](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/DRL_out_of_sample.ipynb) and [TRL_out_of_sample](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/TRL_out_of_sample.ipynb) are for testing the performance of the TRL under different out-of-sample scenarios with bias to intervals (**Figure 4.2**). The training and testing process of network are recored in these files. Please excute the codes in former file first for network offline training and then excute the codes in later file for network performance testing. 
3. The Jupyter Notebook files [DRL_out_of_sample](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/DRL_out_of_sample.ipynb) and [TRL_out_of_sample_result distribution shift](https://github.com/yuht1993/2022.0236/blob/patch-1/exp1_case/distribution%20shift/TRL_out_of_sample_result%20distribution%20shift.ipynb) are for testing the performance of the TRL under different out-of-sample scenarios with distributional shifts (**Figure 4.3**). The training and testing process of network are recored in these files. Please excute the codes in former file first for network offline training and then excute the codes in later file for network performance testing. 

## Numerical Results on Algorithm efficiency and Scability

The _**exp2_case**_ folder contains the data, script and codes for numercail results of _**scaling up instances**_ in Section 4.3 & 4.4.

| 24*20 dimensions | 48*40 dimensions | 72*60 dimensions | 96*80 dimensions | 120*100 dimensions | 144*120 dimensions |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| [Offline_stage_4patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_4patch)  | [Offline_stage_8patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_8patch.ipynb)  | [Offline_stage_12patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_12patch.ipynb) | [Offline_stage_16patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_16patch.ipynb) |[Offline_stage_20patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_20patch.ipynb) | [Offline_stage_24patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Offline_stage_24patch.ipynb) |
| [Test_online_4patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_4patch.ipynb)  | [Test_online_8patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_8patch.ipynb) | [Test_online_12patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_12patch.ipynb) | [Test_online_16patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_16patch.ipynb) | [Test_online_20patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_20patch.ipynb) |[Test_online_24patch](https://github.com/yuht1993/2022.0236/blob/patch-1/exp2_case/Test_online_24patch.ipynb) |

The above Jupyter Notebook files are for testing the performance of the TRL on scaling up instance with factor n=4, 8, 12, 16, 20, 24 (**Table 4.1 and Figure 4.5**). The training and testing process of network are recored in these files. Please excute the codes in **"Offline_stage_xxpatch"** first for network offline training and then excute the codes in **"Test_online_xxpatch"** for network performance testing. 

## Replicating

To replicate the results in [Figure 1](results/mult-test), do either

```
make mult-test
```
or
```
python test.py mult
```
To replicate the results in [Figure 2](results/sum-test), do either

```
make sum-test
```
or
```
python test.py sum
```

## Ongoing Development

This code is being developed on an on-going basis at the author's
[Github site](https://github.com/tkralphs/JoCTemplate).

## Support

For support in using this software, submit an
[issue](https://github.com/tkralphs/JoCTemplate/issues/new).
