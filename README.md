# _CardioFitness_ 🫀💪 — VO2max prediction with wearables

### 📖 Longitudinal cardio-respiratory fitness prediction through wearables in free-living environments 
code for Nature Digital Medicine '22 paper


![header image](https://github.com/sdimi/cardiofitness/blob/main/data/study_overview.png)


<details><summary>Abstract (click to expand)</summary>
<p>

Cardiorespiratory fitness is an established predictor of metabolic disease and mortality. Fitness is directly measured as maximal oxygen consumption (VO2max), or indirectly assessed using heart rate responses to standard exercise tests. However, such testing is costly and burdensome because it requires specialized equipment such as treadmills and oxygen masks, limiting its utility. In this work, we design algorithms and models that convert raw wearable sensor data into cardiorespiratory fitness estimates. We validate these estimates' ability to capture fitness profiles in free-living conditions (N=11,059), along with a longitudinal cohort (N=2,675), and a third external cohort (N=181) who underwent maximal VO2max testing, the gold standard measurement of fitness. Our results show that the combination of wearables and other biomarkers as inputs to neural networks yields a strong correlation to ground truth in a holdout sample (r = 0.82, 95CI 0.80-0.83), outperforming other approaches and models and detects fitness change over time (e.g., after 7 years). We also show how the model's latent space can be used for fitness-aware patient subtyping paving the way to scalable interventions and personalized trial recruitment. These results demonstrate the value of wearables for fitness estimation that today can be measured only with laboratory tests.

</p>
</details>

**This repository**. We provide the code to reproduce the experiments of the paper [1]. We cannot share the entire dataset due to privacy limitations that safeguard health data, however we provide some real data samples (see below). The main input is a chest ECG device which recorded heart rate and movement in 15-second intervals. For the prediction task, the most important outcome is VO2max (cardiorespiratory fitness), measured with a treadmil test. We have pre-processed the data by extracting statistical features. Then we train models using the Fenland I cohort and evaluate them using the Fenland II and BVS cohorts. We provide Python files and Jupyter notebooks with all figures/visualizations included in the paper.

## 🛠️ Requirements
The code is written in python 3.6.0. The main libraries needed to execute the code will be installed through:

    pip install -r requirements.txt
    
You might also need some extra helper libraries like `tqdm` (prettier for-loops) but they are not mandatory.

## 🗂️ Data 
We use data from the [Fenland Study](https://www.mrc-epid.cam.ac.uk/research/studies/fenland/) and the [Biobank Validation Study](https://www.mrc-epid.cam.ac.uk/research/studies/uk-biobank-validation/). We cannot publicly share this data but it is available from the MRC Epidemiology Unit at the University of Cambridge upon reasonable request. To facilitate easier testing of our code, we provide small samples with the same vector shapes and naming conventions. See ``data/extracted_features`` for the features and their order and ``data/vo2max_F1`` for the laboratory treadmill data sample and the data dictionary. Raw sensor signals from a randomly selected participant are provided in ``/data/FL_511496R.dta``.

 
# ▶️ Run
All experiments can be found in python files. Considering that we cannot share the entire datasets but only a single user feature vector, your results will be different than our paper. The hyperparameter tuning was done in a SLURM cluster, and the contribution of the hyperparams was evaluated on the validation set. 

To extract features and train the task 1 models, run: 

    python3 "01_data_extraction.py"
    python3 "02_training_task1.py"
    
To train the task 2 "delta" models, run:

    python3 "02_training_task2_binary.py"
    python3 "02_training_task2_continuous.py"

Last, to evaluate the models on task 3 using new Fenland II sensor data, run:

    python3 "03_inference_task3_FenlandIIsensors.py"


## Pre-trained models

We provide the best model and its weights for Task 1 (called _comprehensive model_ in the paper) in the folder ``/models/20201109-013142``. This model can be used as is to predict VO2max values following the pre-processing pipeline in ``01_data_extraction.py`` and ``02_training_task1.py``.

## How to cite our paper 

Please consider citing our paper if you use code or ideas from this project:

> [1]  Dimitris Spathis, Ignacio Perez-Pozuelo, Tomas I. Gonzales, Yu Wu, Soren Brage, Nicholas Wareham, Cecilia Mascolo. ["Longitudinal cardio-respiratory fitness prediction through wearables in free-living environments."](https://arxiv.org/abs/2205.03116), **Nature Digital Medicine**, 2022 (to appear).

Relevant papers from the same project:

> [2]  Dimitris Spathis, Ignacio Perez-Pozuelo, Soren Brage, Nicholas J. Wareham, Cecilia Mascolo. ["Self-supervised transfer learning of physiological representations from free-living wearable data."](https://dl.acm.org/doi/10.1145/3450439.3451863) In Proceedings of ACM Conference on Health, Inference, and Learning (CHIL), USA, 2021.

> [3] Dimitris Spathis, Ignacio Perez-Pozuelo, Soren Brage, Nicholas J. Wareham, Cecilia Mascolo. ["Learning Generalizable Physiological Representations from Large-scale Wearable Data."](https://arxiv.org/pdf/2011.04601.pdf) In NeurIPS Machine Learning for Mobile Health workshop, Vancouver, Canada, 2020.

> [4] Chi Ian Tang, Ignacio Perez-Pozuelo, Dimitris Spathis, Soren Brage, Nick Wareham, Cecilia Mascolo. ["SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data."](https://dl.acm.org/doi/10.1145/3448112) In Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, Volume 5, Issue 1, 2021.

## License

This code is licensed under the terms and conditions of GPLv3 unless otherwise stated. The actual paper is governed by a separate license and the paper authors retain their respective copyrights.



