import pickle
import pandas as pd

with open('model.pkl', 'rb') as file: 
    model = pickle.load(file) 

test_df = pd.read_csv('test/test.csv')

X_test = test_df.iloc[:, :-1]
y_test = test_df.iloc[:, -1]

score = model.score(X_test, y_test) 
print(f'Test score: {score}') 