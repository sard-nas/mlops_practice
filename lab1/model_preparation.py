import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

train_df = pd.read_csv('train/train.csv')

X_train = train_df.iloc[:, :-1]
y_train = train_df.iloc[:, -1]

model = RandomForestRegressor()
model.fit(X_train, y_train)

with open('model.pkl', 'wb') as file: 
    pickle.dump(model, file)