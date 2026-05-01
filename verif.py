import pandas as pd

df1 = pd.read_csv('imdb_train.csv')
print(f"TRAIN : {len(df1)} lignes")
print(df1['label'].value_counts())
print()

df2 = pd.read_csv('imdb_test.csv')
print(f"TEST : {len(df2)} lignes")
print(df2['label'].value_counts())