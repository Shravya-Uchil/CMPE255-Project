# CMPE255-Project  
\
A network attack could be defined as hacking into a specified target to alter or gain unauthorized access thus making the communicated or stored data packets vulnerable to threats. Network attacks significantly degrade the quality of service experienced by users. Hence a robust security mechanism is necessary to tackle the malicious attacks. The objective of this project is to utilize Data Mining Algorithms and Machine Learning Models to categorize an incoming network traffic as an intrusion or not. With this system in place, if any data with similar characteristics were to arrive in the future, then it can efficiently capture and take the necessary combating actions to deal with the attack. Our model can categorize the data into Normal data or Attack data. Besides, the model can also specify which among the nine attack categories the threat belongs to.  \
### Dataset : https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/ADFA-NB15-Datasets/  
#### Reference : Moustafa, Nour, and Jill Slay. "UNSW-NB15: a comprehensive data set for network intrusion detection systems (UNSW-NB15 network data set)."Military Communications and Information Systems Conference (MilCIS), 2015. IEEE, 2015.  \
## Steps to be Followed:  \
No additional set up needed.\
1. The folder final_version contains the final dataset and implementation\
2. Download the sample dataset. Filename : dataset_final.\
  This is a preprocessed and sampled dataset. The output from this file may differ from what was obtained using the complete dataset\
3. Execute network_anamoly_detection.py file  \\
### Read on for details about other files in this repository\
### Requisites before running Feature_Selector_NCR_SM_Sampler.ipynb (for feature-selector):  
Use the below commands to install feature-selector:\

git clone https://github.com/WillKoehrsen/feature-selector.git  
cd feature-selector  
pip install -e .  

Requirements:  
python==3.6+  
lightgbm==2.1.1  
matplotlib==2.1.2  
seaborn==0.8.1  
numpy==1.14.5  
pandas==0.23.1  
scikit-learn==0.19.1  

Reference : https://github.com/WillKoehrsen/feature-selector  
### Brief description of the project
####
This project implements 6 types of multiclassification models trained with 3 different feature selection methods on pre-processed dataset, evaluated and anlayzed the models results and visualized the same with various metric plots.

### Master Branch: Initial Commits
File Network Anomaly Detection.ipynb	has initial commit of parsing data, structuring and pre-processing .

### feature_importance
1. FeatureSelection_ExtraTreeClassifier.ipynb : This file contains implementation of feature selection using ExtraTreeClassifier model, Undersampling using Randomsampler and oversampling using SMOTE, including all prior data preprocessing.

2. Feature_Selection_RandomForests.ipynb : This file contains implementation of feature selection using RadomTreeClassifier model, Undersampling using Randomsampler and oversampling using SMOTE, including all prior data preprocessing.

3. Feature_Selector_NCR_SM_Sampler.ipynb :  This file contains implementation of feature selection using Feature Selector model, Undersampling using Neighbourhood Cleaning Rule and oversampling using SMOTE, including all prior data preprocessing.

### Models 
1. CNN_ExtratreeClassifier_dataset.ipynb : This file contains implementation of CNN model for classifying attack type categories on attributes selected using Extratreeclassifier along with evaluation, confusion matrix and accuracy plots.

2. CNN_on_ncr_sm.ipynb : This file contains implementation of CNN model for classifying attack type categories on attributes selected using Feature Selector Model along with evaluation, confusion matrix and accuracy plots.

3. ExtraTreeClassifier_KNN_NB_AdaB_XgB_Logistic_ROC_Plots.ipynb : This file contains implementation of KNN, Navie Bayes, AdaBoost, XGBoost, Logistic models for classifying attack type categories on attributes selected using Extratreeclassifier Model along with evaluation, F1 score, Time graph, ROC curve plots.

4. RandomForest_models.ipynb: This file contains implementation of KNN, Navie Bayes, AdaBoost, XGBoost, Logistic, CNN models for classifying attack type categories on attributes selected using Extratreeclassifier Model along with evaluation, confusion matrix,F1 score, Time graph, ROC curve and accuracy plots.


### Evaluation:
Evaluation_Extratreeclassifier_Images : This file contains all Images realted to F1 score, ROC Curve, Accuracy, Time graph, Confusion Matrix, Dataset and Sampling plots where models are trained with Extratreeclassifier features.


