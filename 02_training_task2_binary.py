#author: Dimitris Spathis (ds806@cl.cam.ac.uk)


"""
what: load individual feature CSVs of Fenland's repeats cohort, concat, split to train/test, normalize and do ML
difference from task1: here we focus on the repeats and create three binary tasks based on FII delta

in: CSVs with extracted features, IDs, metadata
out: train/test vectors, figures, and trained models
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.externals import joblib
from sklearn.utils import resample
from sklearn import metrics
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
from keras.utils.np_utils import to_categorical

# FLAGS

#flag to train only with anthro/metadata (True: only antrho variables, False: sensors + anthro)
antrhopometrics = False
#apply PCA and tranform the feature vectors (True: apply, False: use raw data)
pca_flag = True 
#choose between models ('NN': Dense model, 'linear_model': linear)
training_mode = 'NN'
#split data based on outcome ('9010': extreme quantiles 90/10, '8020': moderate quantiles 80/20, 'binary': 50/50)
outcome = '9010' 

# DATA LOADING 

#we load extracted features of the sample user
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


#if one has access to the entire dataset (Fenland I + II), do the following
"""
#Fenland 2 repeats
fitness_df_2 = pd.read_stata("../Fen2_summary_fitness_vars.dta")
fitness_df_2['id'] = fitness_df_2.id[:-2].astype(str).str[:-2]
fitness_df_2 = fitness_df_2[['id', 'TR_FITNESS_HighPt_est',]] 

#we merge the repeats with FI and generate two ID groups so that we can select train/test sets later
repeats = pd.merge(fitness_df_2, frame_with_fitness, on='id', suffixes = ("_repeats","_f1")).dropna(how='any') 
IDs_repeats = frame_with_fitness[frame_with_fitness['id'].isin(repeats.id)].id
IDs_controls = frame_with_fitness[~frame_with_fitness['id'].isin(repeats.id)].id
print (IDs_repeats.shape, IDs_controls.shape) #(2K, 11K)

repeats["fitness_delta"] = repeats.TR_FITNESS_HighPt_est_f1-repeats.TR_FITNESS_HighPt_est_repeats

#split in two disjoint groups of users
#https://stackoverflow.com/questions/44007496/random-sampling-with-pandas-data-frame-disjoint-groups

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=143)

idx1, idx2 = next(gss.split(repeats, groups=repeats.id))
print(idx1.shape, idx2.shape)
    # Get the split.                                           #this iloc :-1 changed to :-2 bc we have two outcomes now (delta)
X_train, y_train = repeats.iloc[idx1,~repeats.columns.str.endswith('_repeats')].iloc[:,:-2],repeats.loc[idx1,'fitness_delta'] 
X_test, y_test = repeats.iloc[idx2,~repeats.columns.str.endswith('_repeats')].iloc[:,:-2], repeats.loc[idx2,'fitness_delta']

if outcome == 'binary':
    y_train = np.where(repeats.loc[idx1,'fitness_delta'] < repeats.loc[idx1,'fitness_delta'].quantile(1/2), 0, 1)
    y_test = np.where(y_test < repeats.loc[idx1,'fitness_delta'].quantile(1/2), 0, 1)
    
elif outcome == '8020':
    y_train = np.where(repeats.loc[idx1,'fitness_delta'] < repeats.loc[idx1,'fitness_delta'].quantile(2/10), 0, 
                       (np.where(repeats.loc[idx1,'fitness_delta'] > repeats.loc[idx1,'fitness_delta'].quantile(8/10), 1, 2)))
    y_test = np.where(y_test < repeats.loc[idx1,'fitness_delta'].quantile(2/10), 0, 
                      (np.where(y_test > repeats.loc[idx1,'fitness_delta'].quantile(8/10), 1, 2)))

elif outcome == '9010':
    y_train = np.where(repeats.loc[idx1,'fitness_delta'] < repeats.loc[idx1,'fitness_delta'].quantile(1/10), 0, 
                       (np.where(repeats.loc[idx1,'fitness_delta'] > repeats.loc[idx1,'fitness_delta'].quantile(9/10), 1, 2)))
    y_test = np.where(y_test < repeats.loc[idx1,'fitness_delta'].quantile(1/10), 0, 
                      (np.where(y_test > repeats.loc[idx1,'fitness_delta'].quantile(9/10), 1, 2)))


#make sure test users do not appear in train set
mask = np.isin(X_test.id, X_train.id)
assert mask.all() == False

if outcome is not "binary":
    #drop the neutral class ==2
    X_train = X_train[(y_train==0) |  (y_train==1)]
    y_train = y_train[(y_train==0) |  (y_train==1)]

    X_test = X_test[(y_test==0) |  (y_test==1)]
    y_test = y_test[(y_test==0) |  (y_test==1)]
"""

#JUST FOR SHOWCASE - BAD PRACTICE!
#in this dummy data release we only use one user, so we'll artificially reuse the same data for both train and test splits
X_train, y_train = frame_with_fitness.iloc[:,:-1],frame_with_fitness.iloc[:,-1]
X_test, y_test = frame_with_fitness.iloc[:,:-1],frame_with_fitness.iloc[:,-1]

#drop the userID (string hence keep only numeric data)
X_train = X_train._get_numeric_data()
X_test = X_test._get_numeric_data()


if antrhopometrics==True:
    X_train =  X_train[[ 'height', 'weight', 'sex', 'age', 'bmi','RHR']]
    X_test =  X_test[[ 'height', 'weight', 'sex', 'age', 'bmi','RHR']]

# Normalize and denoise (PCA) by reusing task1 saved scalers
scaler_name = "data/scaler_FI.save"
scaler = joblib.load(scaler_name)

#scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if pca_flag==True:
    scaler_pca = "data/PCA_FI_mapping_09999.save"
    pca_mapping = joblib.load(scaler_pca)
    pca = pca_mapping
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print ("Explained variance:", pca.explained_variance_ratio_.cumsum()) 

y_train_tensor = to_categorical(y_train)
y_test_tensor = to_categorical(y_test)    
    
if training_mode=='linear_model':
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(class_weight='balanced')
    clf = model.fit(X_train, y_train)
elif training_mode=='NN':
    input_layer = Input(shape=(X_train.shape[1],))    
    x = Dense(128, activation='elu',)(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    final = Dense(2, activation='sigmoid')(x)

    clf = Model(input_layer, final)
    clf.summary()
    clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Create a folder and store the model
    model_time = time.strftime("%Y%m%d-%H%M%S") #use timestamp as folder name...
    path = 'models/%s/'%model_time
    os.makedirs(os.path.dirname('./models/%s/'%model_time))
    # Save architecture for this model
    open('models/%s/model_architecture.json'%model_time, 'w').write(clf.to_json())

    #DON'T CHANGE THE FILEPATH NAMING CONVENTION, THE EVALUATION SCRIPT PARSING DEPENDS ON IT
    filepath="models/%s/weights-regression-improvement-{val_loss:.2f}.hdf5"%model_time
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
       factor=0.1,
       patience=5,
       verbose=1)

    checkpointer = ModelCheckpoint(monitor='val_loss', filepath=filepath, verbose=1, save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')

    history = clf.fit(X_train, y_train_tensor, callbacks=[early_stop,reduce_lr,checkpointer],
                    epochs=500,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.1)
    plt.plot(history.history['val_loss'])

if training_mode=='NN':
    del clf
    folder = model_time # or use another model e.g. '202XXXXX-105719'
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
    
    
predicted = clf.predict(X_test)
#only for NNs
probs = np.argmax(predicted, axis=-1)
print (metrics.accuracy_score(y_test, probs))
print (metrics.roc_auc_score(y_test, predicted[:, 1]))
print (metrics.f1_score(y_test, probs))

print (metrics.confusion_matrix(y_test, probs))
print (metrics.classification_report(y_test, probs))    