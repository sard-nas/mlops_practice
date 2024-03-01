import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

def data_prepocessing(filename):
    df = pd.read_csv(filename)

    df = df.drop(['Id'], axis=1)
    df=df.dropna()

    cat_columns = []
    num_columns = [] 
    for column_name in df.columns:
        if (df[column_name].dtypes == object): 
            cat_columns +=[column_name] 
        else:
            num_columns +=[column_name]

    encoder = OrdinalEncoder()
    df[cat_columns] = encoder.fit_transform(df[cat_columns])

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    df.to_csv(filename)
    
data_prepocessing('train/train.csv')
data_prepocessing('test/test.csv')
