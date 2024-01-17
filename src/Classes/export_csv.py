import pandas as pd
import numpy as np
import pickle

# Load from dql
def load_data_dql(file_path=None):

    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        data = np.array(data)
    
        
    data = {'scores': data[0],
            'apple': data[1],
            'epsilon': data[2],
            'game': np.arange(len(data[0])),
            'steps' : np.cumsum(data[1])
            }
    dataframe = pd.DataFrame(data = data)
    return dataframe

# Load from tql
def load_data_tql(file_path=None):
    
    with open(file_path, 'rb') as file:
            data = pickle.load(file)
            data = np.array(data)

    data = {'scores': data[0], 'apple': data[1]}
    dataframe = pd.DataFrame(data = data)
    return dataframe
    
# Export as csv from pickle
def export_csv(dataframe=None):
    return pd.to_csv(dataframe)

if __name__ == '__main__':
     
    
