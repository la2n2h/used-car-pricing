# used car pricing

## loading data
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

## Identify and handle missing values

replace the "?" symbol with NaN a
```
df.replace('?', np.NaN, inplace = True)
df.head()
```
![image](https://github.com/user-attachments/assets/c76ff6c4-0d5b-4ab5-b69f-627c07e0e276)

### Identify missing values
```
missing_data = df.isnull()
missing_data.head()
```
![image](https://github.com/user-attachments/assets/d5936333-37ee-40a4-ac29-bf814c60f349)

### Count missing value in each column
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


### Deal with missing data

Replace by mean for 
"normalized-losses": 41 missing data, 
"stroke": 4 missing data, 
"bore": 4 missing data, 
"horsepower": 2 missing data, 
"peak-rpm": 2 missing data, 
```
# Calculate the mean value for the "normalized-losses", "stroke", "bore", "horsepower", "peak-rpm" column
avg_norm_loss = df['normalized-losses'].astype('float').mean(axis=0)
print('ave_norm_loss: ' , avg_norm_loss)
avg_stroke = df['stroke'].astype('float').mean(axis=0)
print('avg_stroke: ' , avg_stroke)
avg_bore = df['bore'].astype('float').mean(axis=0)
print('avg_bore: ' , avg_bore)
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print('avg_horsepower: ' , avg_horsepower)
avg_peak_rpm = df['peak-rpm'].astype('float').mean(axis=0)
print('avg_peak_rpm: ' , avg_peak_rpm)

# Replace "NaN" with mean value
df['normalized-losses'] = df['normalized-losses'].fillna(avg_norm_loss)
df['stroke'] = df['stroke'].fillna(avg_stroke)
df['bore'] = df['bore'].fillna(avg_bore)
df['horsepower'] = df['horsepower'].fillna(avg_horsepower)
df['peak-rpm'] = df['peak-rpm'].fillna(avg_peak_rpm)
```
ave_norm_loss:  122.0
avg_stroke:  3.2554228855721394
avg_bore:  3.3297512437810943
avg_horsepower:  104.25615763546797
avg_peak_rpm:  5125.369458128079

Replace by frequency:
"num-of-doors": 2 missing data, replace them with "four".
Reason: 84% sedans are four doors. Since four doors is most frequent, it is most likely to occur
```
# see which values are present in a particular column
df['num-of-doors'].value_counts()
# calculate the most common type
df['num-of-doors'].value_counts().idxmax()
# replace missing data \with "four"
df['num-of-doors'] = df['num-of-doors'].fillna('four')

```
four doors is the most common type. 
![image](https://github.com/user-attachments/assets/7104b49a-c01b-4000-aaef-b34b44ac79b8)
![image](https://github.com/user-attachments/assets/b831d3f0-4268-46e5-9b79-d67b61d41c16)

Drop the whole row:
"price": 4 missing data, simply delete the whole row
Reason: cannot use any data entry without price data for prediction; therefore any row now without price data is not useful.
```
df.dropna(subset = ['price'], axis = 0, inplace = True)
```

reset index, two rows was dropped
```
df.reset_index(drop=True, inplace=True)
```

check data again if any missing value
```
missing_check = df.isnull()
missing_check
for column in missing_check.columns.values.tolist():
    print(column)
    print( missing_check[column].value_counts())
    print('')
```
There is no missing value in the data frame

## Correct data format
check the data type
```
df.dtypes
```
![image](https://github.com/user-attachments/assets/1d96dd07-7522-4412-9dcd-c3621dda234a)

Convert data types to proper format
```
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
```
check data type again
```
df.dtypes
```
![image](https://github.com/user-attachments/assets/dd57b5d8-f13e-4ad5-b9e7-9efcb5500e16)

## Data Standardization
the fuel consumsion of city-mpg and highway-mpg are represented by mpg (miles per gallon) unit. 
Convert fuel consumption with L/100km standard.
```
df['city-L/100km'] = 235/df["city-mpg"]
df['highway-L/100km'] = 235/df["highway-mpg"]
df.head()
```
![image](https://github.com/user-attachments/assets/a1a89edf-d39b-454d-a101-ddb915fffebe)

## Data Normalization
transforming values of several variables into a similar range to scale the columns 'length', ' width', 'height'
```
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()
df['height'] = df['height']/df['height'].max()
df[["length","width","height"]].head()
```
![image](https://github.com/user-attachments/assets/3ba0f750-ba5d-43d7-b77b-6db2d960a2ac)

## Binning
transforming continuous numerical variables into discrete categorical 'bins' for grouped analysis.
In the data frame, "horsepower" is a real valued variable ranging from 48 to 288 and it has 59 unique values. Rearrange them into three ‘bins' high horsepower, medium horsepower, and little horsepower (3 types) for simplify analysis.
```
df["horsepower"]=df["horsepower"].astype(int, copy=True)
```
see the distribution of horsepower
```
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
```
![image](https://github.com/user-attachments/assets/080b2d95-1f72-442e-afd0-005aee46f758)

Build a bin array with a minimum value to a maximum value by using the bandwidth calculated above. The values will determine when one bin ends and another begins.
```
bins =np.linspace(min(df['horsepower']), max(df['horsepower']),4)
bins
```
![image](https://github.com/user-attachments/assets/67432b8a-3f60-414c-9713-4fb11df04535)

Set group names:
```
group_names = ['Low', 'Medium', 'High']
```
apply the function 'cut' to determine what each value of df['horsepower'] belong to
```
df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels = group_names, included_lowest = True)
df[['horsepower', 'horsepower-binned']].head()
```
![image](https://github.com/user-attachments/assets/f8a93aee-aab1-44dd-b308-4ba922b62a9c)

```
df['horsepower-binned'].value_counts()
```
![image](https://github.com/user-attachments/assets/cbcd9fcc-20e6-4ccf-a0da-a00a71ae3144)

Plot the distribution of each bin
```
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
```
![image](https://github.com/user-attachments/assets/3fb24fe9-d26a-4e6d-9589-a0d43273d206)
use a histogram to visualize the distribution of bins we created above
```
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
```
![image](https://github.com/user-attachments/assets/41ca2949-8865-4194-9318-32c0b52474ec)

## Indicator Variable
#### The column "fuel-type" has two unique values: "gas" or "diesel".  convert "fuel-type" to indicator variables to use this attribute in regression analysis.
Get the indicator variables and assign it to data frame "dummy_variable_1"
```
dummy_variable_1 = pd.get_dummies(df['fuel-type']) # 
dummy_variable_1.head()
```
![image](https://github.com/user-attachments/assets/07228253-9c1c-453a-ad2f-76a26320492b)
Change the column names for clarity
```
dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()
```
```
df = pd.concat([df, dummy_variable_1], axis=1)
df.head()
```
![image](https://github.com/user-attachments/assets/9748447c-e557-422c-b164-7d207cf3b411)

#### The column "aspiration" has two unique values: "std" or "turbo".  convert "aspiration" to indicator variables to use this attribute in regression analysis.
Get the indicator variables and assign it to data frame "dummy_variable_2"
```
dummy_variable_2 = pd.get_dummies(df['aspiration']) # 
dummy_variable_2.head()
```
![image](https://github.com/user-attachments/assets/07228253-9c1c-453a-ad2f-76a26320492b)
Change the column names for clarity
```
dummy_variable_2.rename(columns={'std':'aspiration-std', 'turbo':'aspiration-turbo'}, inplace=True)
df = pd.concat([df, dummy_variable_2], axis=1)
df.head()
```

```
df = pd.concat([df, dummy_variable_2], axis=1)
df.head()
```
![image](https://github.com/user-attachments/assets/cec46071-236c-4bf2-9586-e1cfdc2adaae)
```
df = df.loc[:, ~df.columns.duplicated()]  # Giữ lại duy nhất một cột mỗi loại
```
![image](https://github.com/user-attachments/assets/fdc605ca-2bfd-451f-85b4-dbdbe4e97e10)

## save the data after clean
```
df.to_csv('D:/Data anlysis- working sheet/python/data/auto_clean_df.csv')
```





