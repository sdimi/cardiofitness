#author: Dimitris Spathis (ds806@cl.cam.ac.uk)


"""
what: load individual feature CSVs of Fenland's full cohort, concat, split to train/test, normalize and do ML

in: CSVs with extracted features, IDs, metadata
out: train/test vectors, figures, and trained models
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

X_train, y_train = frame_with_fitness[frame_with_fitness.index.isin(IDs_controls.index.values)].iloc[:,:-1],frame_with_fitness[frame_with_fitness.index.isin(IDs_controls.index.values)].iloc[:,-1]
X_test, y_test = frame_with_fitness[frame_with_fitness.index.isin(IDs_repeats.index.values)].iloc[:,:-1],frame_with_fitness[frame_with_fitness.index.isin(IDs_repeats.index.values)].iloc[:,-1]

#make sure train/test UserIDs do not overlaps
mask = np.isin(X_test.id, X_train.id)
assert mask.all() == False
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

# Normalize and denoise (PCA)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

if pca_flag==True:
    pca_explained = 0.9999
    pca = PCA(pca_explained) 
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    print ("Explained variance:", pca.explained_variance_ratio_.cumsum()) 
    plt.plot(pca.explained_variance_ratio_.cumsum())

#save both scaler and PCA mapping for reusing in other tasks
from sklearn.externals import joblib
pca_filename = "data/PCA_FI_mapping_09999.save"
joblib.dump(pca, pca_filename) 

scaler_filename = "data/scaler_FI.save"
joblib.dump(scaler, scaler_filename) 

if training_mode=='linear_model':
    model = LinearRegression()
    clf = model.fit(X_train, y_train)
    print (clf)
    print ("saving model..")
    joblib.dump(clf, './models/linear_model_FI.pkl') 
    
elif training_mode=='NN':
    input_layer = Input(shape=(X_train.shape[1],))    
    x = Dense(128, activation='elu',)(input_layer)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='elu',)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    final = Dense(1)(x)

    clf = Model(input_layer, final)
    clf.summary()
    clf.compile(optimizer='adam', loss='mse')
    
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

    history = clf.fit(X_train, y_train.values, callbacks=[early_stop,reduce_lr,checkpointer],
                    epochs=500,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.1)
    plt.plot(history.history['val_loss'])

if training_mode=='NN':
    #load best model (trained on all data, not only this sample from this release)
    folder =  '20201109-013142'
    clf = model_from_json(open('./models/'+ folder +'/model_architecture.json').read())
    files = glob.glob('./models/'+folder+'/*.hdf5')
    #WARNING! DEPENDS ON THE NAMING CONVENTION!
    #we parse the filename ('./models/20191029-125927/weights-regression-improvement-51.03.hdf5')
    #and we convert the MSE to a float in order to sort, the first is the lowest (lowest val_loss)
    weights = sorted(files, key=lambda name: float(name[56:-5]))[0]
    print ("=============Best model loaded:", weights)
    clf.load_weights(weights)   
    clf.compile(loss="mse", optimizer="adam")
    clf.summary()
elif training_mode== 'linear_model':
    linear_model = "models/linear_model_FI.pkl"
    clf = joblib.load(linear_model)    

def evaluate (X,y, ml_mode=True):
    #evaluation metrics
    if ml_mode:
        predicted = np.squeeze(clf.predict(X))
        print (clf)
    else:
        predicted = X
    mse, rmse, mae, std_mae, r2, mape, corr =  error_metrics(y.astype('float').values,predicted)
    return predicted

predicted_test = evaluate(X_test, y_test, )

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