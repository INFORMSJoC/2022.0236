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

This repository includes the source code and computational results of transfer reinforcement learning approach for randomly generated numerical instances presented in the paper. The codes and random instance generations work in a Miniforge3 environment with Python 3.9 in a M1 mac platform.

## Data

This directory contains the values of SEAIDR model parameters ('para_truth_xx' and 'out_of_sample_xx' files) used in the numerical experienments in Section 4. 
To open these files, try to use:

```
import pickle
pickle.load(open('para_truth_xx', 'rb'))
```

The .csv file records the initial state of SEAIDR model used in the scaling up instances.

## Results

### Numerical Results on Solution Optimality and Robustness

The Jupyter Notebook files **DRL_uncertainty_result** and **TRL_uncertainty_result** records the training and testing processes of neural networks and presents the performance of the TRL under different uncertainty intervals (**Figure 4.1 and Figure 4.4**). 

The Jupyter Notebook files **DRL_out_of_sample** and **TRL_out_of_sample** records the training and testing processes of neural networks and presents the performance of the TRL under different out-of-sample scenarios with bias to intervals (**Figure 4.2**). 

The Jupyter Notebook files **DRL_out_of_sample** and **TRL_out_of_sample_result distribution shift**  records the training and testing processes of neural networks and presents the performance of the TRL under different out-of-sample scenarios with distributional shifts (**Figure 4.3**). 

### Numerical Results on Algorithm efficiency and Scability

The following Jupyter Notebook files: 

| 24*20 dimensions | 48*40 dimensions | 72*60 dimensions | 96*80 dimensions | 120*100 dimensions | 144*120 dimensions |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Offline_stage_4patch  | Offline_stage_8patch  | Offline_stage_12patch | Offline_stage_16patch |Offline_stage_20patch | Offline_stage_24patch |
| Test_online_4patch  | Test_online_8patch | Test_online_12patch | Test_online_16patch | Test_online_20patch |Test_online_24patch |

record the training and testing processes of neural networks and presents the TRL results on **scaling up instances** with factor n=4, 8, 12, 16, 20, 24 (**Table 4.1 and Figure 4.5**)

### Application to a Pandemic Control Case

The Jupyter Notebook files: **Offline_stage_high**,  **Test_online_high**, **Offline_stage_low**, and **Test_online_low**  record the training and testing processes of neural networks and presents the TRL results under high-level and low-level budget scenarios (**Table 5.1**).

## Scripts

This directory includes the scripts for training and testing the neural network for TRL implementation. Due to file upload size limitations, we have stored the initialization and training results of neural network weights on a cloud drive. Please refer to the Readme file in the **network initialization** and **training result** folders in exp1_case, exp2_case, and exp3_case for details about these results. For any issue about the cloud drive please contact [yuht@dlut.edu.cn](yuht@dlut.edu.cn)
