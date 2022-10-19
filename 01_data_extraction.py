#author: Dimitris Spathis (ds806@cl.cam.ac.uk)

"""
what: pre-process acceleration and heart-rate data of Fenland I full cohort:

in: user IDs, ACC, HR
out: CSVs of extracted features + metadata per user
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import glob

#we use a sample user for this release
user_id = '511496R'

#if one has access to the entire dataset, do the following
"""
file_list =glob.glob('../Actiheart/*.dta') #list of all .dta files
user_id = [y[53:-4] for y in file_list] #userID stripped out of files
print ("number of users Fenland I:",len(user_id))
"""

users = pd.DataFrame(user_id)
users.columns = ['id']
fitness_df = pd.read_csv("/data/treadmill_HR.csv")
fitness_fenland = pd.merge(users,fitness_df, left_on =['id'],right_on=['SerNo'], how = 'inner')
fitness_fenland_HR = fitness_fenland[['P_rest_HRaS_min_4','P_rest_HRaS_min_5','P_rest_HRaS_min_6','id']]
#average minutes of treadmill HR
fitness_fenland["mean_rest_HR"] = fitness_fenland[['P_rest_HRaS_min_4','P_rest_HRaS_min_5','P_rest_HRaS_min_6']].mean(axis=1) #Resing heart rate is calculated through sleeping heart rate and resting treadmill HR
fitness_fenland["RHR"] = fitness_fenland["mean_rest_HR"] + fitness_fenland["P_SHR"]
fitness_fenland_HR = fitness_fenland[["RHR",'id']] 

fitness_fenland_HR.dropna(how='any', inplace=True)
assert fitness_fenland_HR.isnull().any().any() == False
print ("Number of users after merging with treadmill data for RHR:",fitness_fenland_HR.shape[0])

def feature_extraction(array, feature_no): #3D tensor -> 2D features extraction from every [:,:,d] dimension
    array = pd.DataFrame(array)

    #Pandas .describe() features (count mean	std	min	25%	50%	75%	max)
    array_features = array.describe().T.add_suffix('_%s'%feature_no)
    
    #we use the slope of Linear Regression as a feature
    transp_array = array
    array_features["slope_%s"%feature_no] = transp_array.apply(lambda x: np.polyfit(transp_array.index, x, 1)[0])
    
    return array_features

for i in tqdm(range(len(user_id))):       
    #loop through all users (takes XX hours)
    try:
        uID = user_id[i] #current user
        file = pd.read_stata("/data/*.dta") #read file
        file = pd.merge(file,fitness_fenland_HR[fitness_fenland_HR.id==uID]) #merge with demographics   
        assert file.shape[0] > 3600 #select users with more than 3.5 days of data (3600 minutes)
    except (FileNotFoundError,AssertionError) as e:
        print("exception:", uID)
    else:
        print("userid:", uID)
        file = file[file.PWEAR>0] #remove buffer no_wear
        file['real_time'] = pd.to_datetime(file['real_time'], dayfirst=True) #datetime pandas format
        file.index = pd.to_datetime(file['real_time'])
        file['ENMO'] = (file["ACC"]/0.0060321) + 0.057
        file = file[file.PWEAR>0] #remove buffer heart rate
        #calculate HRV from IBI (data is corrupted so we should flag/not use that datapoint)
        file['hrv_milliseconds'] = np.where(file.min_ibi_2_in_milliseconds != 1992.0,
                                          file['max_ibi_2_in_milliseconds']-file['min_ibi_1_in_milliseconds'], np.nan)
        file.fillna(file.mean(), inplace=True) #fill nans with mean and if completely empty fill with 0
        file.fillna(0, inplace=True)
        
        file["month"] = pd.to_datetime(file["real_time"]).apply(lambda x: x.month)
        months_in_year = 12
        file['month_sin_time'] = np.sin(2*np.pi*file["month"]/months_in_year)
        file['month_cos_time'] = np.cos(2*np.pi*file["month"]/months_in_year)
        
        #non-cal MVPA (using only acceleremeter data)
        file['Sed_noncal'] = file['ACC'].apply(lambda x: x if x < 1 else 0)
        file['MVPA_noncal'] = file['ACC'].apply(lambda x: x if x >= 1 else 0)
        file['VPA_noncal'] = file['ACC'].apply(lambda x: x if x >= 4.15 else 0)
        
        # MET related calculations
        file['MET_sed'] = file['stdMET_highIC_Branch'].apply(lambda x: x if x <= 0.5 else 0)
        file['MET_LPA'] = file['stdMET_highIC_Branch'].apply(lambda x: x if x > 1.5 else 0)
        file['MET_MVPA'] = file['stdMET_highIC_Branch'].apply(lambda x: x if x > 2 else 0)
        file['MET_VigPA'] = file['stdMET_highIC_Branch'].apply(lambda x: x if x > 5 else 0)
        
        #find minutes where MET > or < of X, then count their daily instances and average
        file['sed_daily_count'] = file[file.MET_sed.le(2)==True].resample('D').count().mean().fillna(0)[0]
        file['mvpa_daily_count'] = file[file.MET_MVPA.between(2, 5)==True].resample('D').count().mean().fillna(0)[0]
        file['vpa_daily_count'] = file[file.MET_VigPA.ge(5)==True].resample('D').count().mean().fillna(0)[0]
        
        #find noncal minutes where MET > or < of X, then count their daily instances and average
        file['sed_daily_count_noncal'] = file[file.Sed_noncal.le(1)==True].resample('D').count().mean().fillna(0)[0]
        file['mvpa_daily_count_noncal'] = file[file.MVPA_noncal.between(1, 4.15)==True].resample('D').count().mean().fillna(0)[0]
        file['vpa_daily_count_noncal'] = file[file.VPA_noncal.ge(4.15)==True].resample('D').count().mean().fillna(0)[0]
        
        #calculate BMI
        file["bmi"] = file["weight"] / (file["height"] * file["height"])
        
        file_variables = file[['id','height','month', 'month_sin_time',
                               'month_cos_time', 'weight', 'sex', 'age', 'bmi', 'RHR',
                              'sed_daily_count','mvpa_daily_count','vpa_daily_count',
                              'sed_daily_count_noncal','mvpa_daily_count_noncal','vpa_daily_count_noncal']]
        file_sensors = file[['ACC','mean_hr','hrv_milliseconds','ENMO', 'MET_sed',
                             'MET_MVPA','MET_LPA', 'MET_VigPA','Sed_noncal','MVPA_noncal','VPA_noncal']]
        
        #extract statistical features for each timeseries and concatenate into one array
        all_df = []
        for z,j in enumerate(file_sensors.columns.values): #loop in sensor number and name
            all_df.append(feature_extraction(file_sensors.iloc[:, [z]].values,j))
        print ("Concatenating all features in one array..")
        ts_features = pd.concat(all_df,axis=1)  
        
        #combine the first line of metadata (fixed across time) with the single line of week level sensor features
        merged = file_variables.head(1).reset_index().append(ts_features,ignore_index=True)
        merged = merged.apply(lambda x: pd.Series(x.dropna().values)) #hacky way to merge two single-row dataframes
        
        assert merged.isna().sum().sum() == 0 #then, make sure that no nans exist [TEST]
        
        merged.to_csv("/data/"+uID+"_features.csv")