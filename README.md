# used car pricing

# loading data
Load the dataset to a pandas dataframe named 'df'
```
import pandas as pd
import numpy as np
```

```
file_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
```

```
df = pd.read_csv(file_path, header = None)
df.head()
```
![image](https://github.com/user-attachments/assets/f69147fd-101f-46d1-a590-0881c4ec13fc)

change the header names
```
df.columns = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
df.head()
```
![image](https://github.com/user-attachments/assets/46fac347-77af-4769-85da-77708fae64f8)

Save Dataset
```
df.to_csv('D:/Data anlysis- working sheet/python/data/auto.csv', index=False)
```
Basic Insights from the Data set
```
df.describe(include = 'all')
```
![image](https://github.com/user-attachments/assets/a776ea70-bab7-429b-8993-e37eced3a467)

```
df.info()
```
![image](https://github.com/user-attachments/assets/88ca608f-14fa-4bc7-aeb2-bd14fb263d28)

# Pre-processing data

replace the "?" symbol with NaN a
```
df.replace('?', np.NaN, inplace = True)
df.head()
```
![image](https://github.com/user-attachments/assets/c76ff6c4-0d5b-4ab5-b69f-627c07e0e276)

Identify missing values
```
missing_data = df.isnull()
missing_data.head()
```
![image](https://github.com/user-attachments/assets/d5936333-37ee-40a4-ac29-bf814c60f349)

Count missing value in each column
```
for column in missing_data.columns.values.tolist():
    print(column)
    print( missing_data[column].value_counts())
    print('')
```
Based on the summary each column has 205 rows of data and seven of the columns containing missing data:
"normalized-losses": 41 missing data,
"num-of-doors": 2 missing data,
"bore": 4 missing data,
"stroke" : 4 missing data,
"horsepower": 2 missing data,
"peak-rpm": 2 missing data,
"price": 4 missing data,


Deal with missing data

Replace by mean for 
"normalized-losses": 41 missing data, 
"stroke": 4 missing data, 
"bore": 4 missing data, 
"horsepower": 2 missing data, 
"peak-rpm": 2 missing data, 
```
# Calculate the mean value for the "normalized-losses" column
avg_norm_loss = df['normalized-losses'].astype('float').mean(axis=0)
print('ave_norm_loss: ' , avg_norm_loss)
# Calculate the mean value for the "stroke" column


```

Replace by frequency:
"num-of-doors": 2 missing data, replace them with "four".
Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur
```
```

Drop the whole row:
"price": 4 missing data, simply delete the whole row
Reason: You want to predict price. You cannot use any data entry without price data for prediction; therefore any row now without price data is not useful to you.
```
```





