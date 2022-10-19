#author: Dimitris Spathis (ds806@cl.cam.ac.uk)

import keras
import keras.backend as K
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score, mean_absolute_percentage_error
from math import sqrt
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def error_metrics(test, predicted):
    mse =  mean_squared_error(test, predicted) #MSE
    rmse =  sqrt(mean_squared_error(test, predicted)) #RMSE
    mae = mean_absolute_error(test, predicted) #MAE
    std_mae = np.std(np.abs(test - predicted))
    r2 = r2_score(test, predicted) #rˆ2
    mape = mean_absolute_percentage_error(test, predicted)
    corr = np.corrcoef(test, predicted)[0][1]

    return mse, rmse, mae, std_mae, r2, mape, corr

class PlotLosses(keras.callbacks.Callback): #live updating plot with loss and validation loss
    
    def __init__(self, model_time):
        self.model_time = model_time #do this function in order to pass the model folder for the saved png
        
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        
        #clear_output(wait=True)
        plt.clf() #new addition, important if not in jupyter, equivalent to clear_output(wait=True)
        
        plt.plot(self.x, self.losses, label="train")
        plt.plot(self.x, self.val_losses, label="val")
        plt.ylabel('Loss')
        plt.legend()        
        plt.xlabel('Epoch')
        plt.savefig("models/%s/training_curves.png"%self.model_time, bbox_inches="tight")       
        #plt.show();