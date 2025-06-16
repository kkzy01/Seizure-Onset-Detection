# Precise Seizure Onset Detection of EEG Signals Using Machine Learning Models

## Leveraging three models to perform precise onset detection on EEG signals.

- Baseline model [One-Class Novelty Detection for Seizure Analysis from Intracranial EEG](https://www.jmlr.org/papers/v7/gardner06a.html)
- SPaRCNet [Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns During EEG Interpretation](https://pubmed.ncbi.nlm.nih.gov/36878708/)
- DSOSD
![image](https://github.com/user-attachments/assets/584a2a17-d77c-42dd-a2d2-7b597ebf91a6)
![image](https://github.com/user-attachments/assets/9166c896-74dc-4436-9226-30daab3ed35e)



## instructions:

- Use python>=3.8
- install requirements.txt
- change iEEG username and password path, and change file path
- run example.ipynb for three different models


## Future directions:
- set those with latency > /threshold/ as miss detection, and then analyze the performance
- hyperparameter tuning (threshold, window lengths, etc.)
- try on longer unclipped signals
