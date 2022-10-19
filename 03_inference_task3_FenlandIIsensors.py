#author: Dimitris Spathis (ds806@cl.cam.ac.uk)

"""
what: load individual feature CSVs of Fenland II repeats (sensors), concat, normalize and do ML
difference from other tasks: here we don't train models, we just predict using the new sensor data using Task 1's model

in: CSVs with extracted features, IDs, metadata
out: train/test vectors and trained models
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.utils import resample
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

#nice figure colors
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71","#f4cae4"]
import seaborn as sns
sns.set_context("poster")

import glob
from utils import *
import time
import os

import keras
from keras.layers import Input, Dense, Dropout
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint


# FLAGS

#flag to train only with anthro/metadata (True: only antrho variables, False: sensors + anthro)
antrhopometrics = False
#apply PCA and tranform the feature vectors (True: apply, False: use raw data)
pca_flag = True 
#choose between models ('NN': Dense model, 'linear_model': linear)
training_mode = 'NN'

# DATA LOADING 

#we load extracted features of the sample user (for this showcase we will reuse the same user as previously, but if one has access to FII sensor data, it should be loaded here)
#to extract these features we use a similar script to 01_data_extraction.py for FII sensor data
frame = pd.read_csv("data/511496R_features.csv", index_col=0)
#load Vo2max data
fitness_df = pd.read_csv("/data/vo2max_F1.csv")
fitness_df.index = fitness_df['serno'] #move the user_id to index
del fitness_df['serno']
fitness_df = fitness_df[['P_TR_FITNESS_HighPt_est']] #standartize column names with Fenland II cohort
fitness_df.rename(columns={'P_TR_FITNESS_HighPt_est': 'TR_FITNESS_HighPt_est'}, inplace=True)

#merge Fenland 1 with sensors and remove missing fitness entries
frame_with_fitness =  pd.merge(frame, fitness_df, left_on='id', right_on='serno').dropna(how='any') 

#remove timestamp, count_ features and index (the feature extraction created an additional index column)
frame_with_fitness = frame_with_fitness.loc[:,~frame_with_fitness.columns.str.startswith('real_')]
frame_with_fitness = frame_with_fitness.loc[:,~frame_with_fitness.columns.str.startswith('count_')]
frame_with_fitness = frame_with_fitness.loc[:,~frame_with_fitness.columns.str.startswith('index')]

#remove treadmill informed MVPA variables
frame_with_fitness = frame_with_fitness.loc[:,~frame_with_fitness.columns.str.contains('_MET_')]
frame_with_fitness = frame_with_fitness.loc[:,~frame_with_fitness.columns.str.endswith('daily_count')]

#if one has access to the entire dataset, do the following
"""
#Fenland 2 repeats
fitness_df_2 = pd.read_stata("../Fen2_summary_fitness_vars.dta")
fitness_df_2['id'] = fitness_df_2.id[:-2].astype(str).str[:-2]
fitness_df_2["bmi"] = fitness_df_2["weight"] / (fitness_df_2["height"] * fitness_df_2["height"])
fitness_df_2 = fitness_df_2[['id','age', 'bmi', 'height', 'weight','TR_FITNESS_HighPt_est']] 

#we merge the repeats with FI and generate two ID groups so that we can select train/test sets later
#we reset the index because it was creating issues in disjoint splits later
repeats = pd.merge(frame_with_fitness, fitness_df_2, on='id', suffixes = ("_f1","_repeats")).dropna(how='any').reset_index(drop=True)
IDs_repeats = frame_with_fitness[frame_with_fitness['id'].isin(repeats.id)].id
IDs_controls = frame_with_fitness[~frame_with_fitness['id'].isin(repeats.id)].id
repeats = repeats.reindex(sorted(repeats.columns), axis=1) #IMPORTANT, we have to reorder columns alphabetically to match with FI data

#we will only do inference here so we only require X, y (no splits)
my_suffixes = ("_f1", "HighPt_est_repeats") #remove both F1 demographics and label at once
X, y,  = repeats.loc[:,~repeats.columns.str.endswith(my_suffixes)],repeats.loc[:,'TR_FITNESS_HighPt_est_repeats'],
X = X.reindex(sorted(X.columns), axis=1) #IMPORTANT, we have to reorder columns alphabetically to match with FI data
X = X._get_numeric_data()
"""

#JUST FOR SHOWCASE!
#in this dummy data release we only use one user, so we'll artificially reuse the same data for FII inference
my_suffixes = ("_f1", "HighPt_est_repeats") #remove both F1 demographics and label at once
X, y = frame_with_fitness.loc[:,~frame_with_fitness.columns.str.endswith(my_suffixes)],frame_with_fitness.loc[:,'TR_FITNESS_HighPt_est_repeats']
X = X.reindex(sorted(X.columns), axis=1) #IMPORTANT, we have to reorder columns alphabetically to match with FI data
X = X._get_numeric_data()

if antrhopometrics==True:
    X_train = X_train[[ 'height_f1', 'weight_f1', 'sex', 'age_f1', 'bmi_f1','RHR']] #'RHR'
    X_test = X_test[[ 'height_f1', 'weight_f1', 'sex', 'age_f1', 'bmi_f1','RHR']] #'RHR'
    print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)    


scaler_name = "data/scaler_FI.save"
scaler = joblib.load(scaler_name)
X = scaler.transform(X)

if pca_flag==True:
    scaler_pca = "data/PCA_FI_mapping_09999.save"
    pca_mapping = joblib.load(scaler_pca)
    pca = pca_mapping
    X = pca.transform(X)
    print ("Explained variance:", pca.explained_variance_ratio_.cumsum()) 

if training_mode=='NN':
    folder = '20201109-013142'
    #load best model from the folder
    clf = model_from_json(open('./models/'+ folder +'/model_architecture.json').read())
    files = glob.glob('./models/'+folder+'/*.hdf5')
    #WARNING! DEPENDS ON THE NAMING CONVENTION!
    #we parse the filename ('./models/20191029-125927/weights-regression-improvement-51.03.hdf5')
    #and we convert the MSE to a float in order to sort, the first is the lowest (lowest val_loss)
    weights = sorted(files, key=lambda name: float(name[56:-5]))[0]
    print ("=============Best model loaded:", weights)
    clf.load_weights(weights)   
    clf.compile(loss="mse", optimizer="adam")
    
def evaluate (X,y, ml_mode=True):
    #evaluation metrics
    if ml_mode:
        predicted = np.squeeze(clf.predict(X))
        print (clf)
    else:
        predicted = X
    mse, rmse, mae, std_mae, r2, mape, corr =  error_metrics(y.astype('float').values,predicted)
    return predicted

predicted_test = evaluate(X_test, y_test)

def CI95(predicted, y_test):
    #calculate confidence intervals through bootstrapping 
    print ("Data size:", len(y_test))
    from sklearn.utils import resample
    bootstrap=[]
    for i in tqdm(range(500)): 
        pairs = np.array(resample(predicted, y_test.astype('float'), replace=True)) #resample in pairs of (pred,y)
        pred , y = pairs[0,:], pairs[1,:] #select the data
        mse, rmse, mae, std_mae, r2, mape, corr =  error_metrics(y,pred)
        bootstrap.append([mse, rmse, mae, std_mae, r2, mape, corr]) #calculate the error and add to the list of boostrap iterations
    return bootstrap

bootstrap = CI95(predicted_test, y_test)

def CI95andMean (bootstrap, y_test, y_pred):
    #put CIs and mean values to the same dataframe (where 0.000 column -> mean value)
    final_metrics = pd.DataFrame(np.array(bootstrap))
    mean_values = pd.DataFrame(np.array(error_metrics(y_test.astype('float'),y_pred)))
    final_metrics.columns = ['mse', 'rmse', 'mae','std_mae', 'r2', 'mape', 'corr']
    display(pd.concat([final_metrics.quantile(0.025), final_metrics.quantile(0.975), 
               mean_values.set_index(final_metrics.quantile(0.025).index)],axis=1))

CI95andMean(bootstrap, y_test, predicted_test)

#put results into a dataframe with predictions and true values
predicted_vs_truth = pd.DataFrame(np.column_stack((predicted_test,y, repeats.id )), columns=['predicted', 'truth', 'id'])
print (predicted_vs_truth)
predicted_vs_truth.to_csv("data/predictions_FenlandII_labels_userID.csv")