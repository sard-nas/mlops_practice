import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split

if (not os.path.exists('./dataset.csv')):
    url = 'https://github.com/ASWINBABUKV/House-Price-Prediction/raw/main/housepriceprediction.csv'
    response = requests.get(url)

    with open('dataset.csv', 'wb') as file:
        file.write(response.content)

df = pd.read_csv('dataset.csv')

train, test = train_test_split(df, test_size=0.2, random_state=73)

train.to_csv('train/train.csv', index=False)
test.to_csv('test/test.csv', index=False)