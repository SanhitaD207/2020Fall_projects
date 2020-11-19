import pandas as pd

data1 = pd.read_excel("MVCollisionsDataDictionary_20190813_ERD.xlsx", skiprows=15)

print(data1[0,:])