import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import pickle as pkl
import time

class PreProcessing():

    def __init__(self, 
                    dataset_url,
                    dataset_name
                ):
        self.dataset_url = dataset_url 
        self.dataset_name = dataset_name 

    
    def preprocess_dataset(self):
        
        starttime = time.time()
        
        data = pd.read_csv(self.dataset_url, sep=';', nrows=None)

        column = ['SERVICE', 'PROTOCOL', 'DESTPORT']
        
        df = data[column]
        
        df['PROTOCOL'] = df['PROTOCOL'].apply(lambda row: 1 if row == 'TCP' else 0) 
        
        X_cat = df
            
        onehotencoder = OneHotEncoder()
        X_cat = onehotencoder.fit_transform(X_cat[['SERVICE']]).toarray()
        X_cat = pd.DataFrame(X_cat)
        
        df_ = pd.concat([df, X_cat], axis=1).drop('SERVICE', axis=1)
        
        with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'data', '%s_preprocessing_syntheticdata.pkl' % ( self.dataset_name ))), 'wb') as f:
            df_.to_pickle(f)
        
        df__train, df__test = train_test_split(df_, test_size=0.25, random_state=42)
        
        path_df_train_save = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'data', '{}_processed_train.csv'.format(self.dataset_name)))  
        path_df_test_save = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'data', '{}_processed_test.csv'.format(self.dataset_name)))  
    
        df__train.to_csv(path_df_train_save, index=False)
        df__test.to_csv(path_df_test_save, index=False)

        endtime = time.time()
        elapsed_time = endtime - starttime

        print(self.dataset_name,'\nElapsed Time:', elapsed_time, 'seconds\n')

        exectime = {
                        'starttime' : starttime,
                        'endtime' : endtime,
                        'time' : elapsed_time
                    }

        with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'data', '%s_executiontime_preprocessing_syntheticdata.pkl' % ( self.dataset_name ))), 'wb') as f:
                        pkl.dump(exectime, f)        
        
if __name__ == "__main__":
    
    url_local = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, 'data', 'traffic_table.csv'))
    pre_proc_local = PreProcessing(url_local, 'local')
    pre_proc_local.preprocess_dataset()