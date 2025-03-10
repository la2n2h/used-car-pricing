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

# Data analysis

## load the cleaned data
```
df = pd.read_csv('D:/Data anlysis- working sheet/python/data/auto_clean_df.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/a4db4f6a-0b9e-4840-8f19-7898f6feaa7b)

## Analyzing Individual Feature Patterns Using Visualization
import visualization packaged
```
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```
print dataframe type for understand the of variable which are dealing with
```
df.dtypes
```
![image](https://github.com/user-attachments/assets/5b48693f-475c-4f3e-a59c-8152b0e86f57)

#### calculate the correlation between variables of type "int64" or "float64"
calculate the correlation between variables of type "int64" or "float64" using the method "corr" to: Identify Relationships Between Variables, Feature Selection for Machine Learning, Detect Multicollinearity.
Coefficient = 1 → The two variables have a perfect correlation (both increase or both decrease).
Coefficient close to 1 (e.g., 0.95) → The two variables have a strong positive correlation (one increases, the other also increases).
Coefficient close to 0 (e.g., 0.05) → The two variables are not related to each other.
Coefficient close to -1 (e.g., -0.85) → The two variables have a strong negative correlation (one increases, the other decreases).
(The diagonal elements are always one)
```
numeric_df = df.select_dtypes(include=['float64', 'int64'])
numeric_df.corr()
```
![image](https://github.com/user-attachments/assets/04721294-6232-4697-9795-43e60fc90d2c)

Find the correlation between the following columns: bore, stroke, compression-ratio, and horsepower.
```
df[['bore','stroke','compression-ratio','horsepower']].corr()
```
![image](https://github.com/user-attachments/assets/4e0e8fb8-a777-4d81-a7af-e3f582c86891)
bore has a significant impact on horsepower, so increasing the bore diameter can help increase engine power.
compression-ratio has a slight but negative effect on horsepower, meaning adjustments to the compression ratio should be carefully considered.
stroke does not have a meaningful impact on horsepower, indicating that piston stroke length is not a key factor in determining engine power.

#### deal with Continuous Numerical Variables
In order to start understanding the (linear) relationship between an individual variable and the price, use "regplot" which plots the scatterplot plus the fitted regression line for the data

the (linear) relationship between an 'engine-size' and the'price'
```
sns.regplot(x='engine-size', y='price', data = df)
plt.ylim(0,)
df[["engine-size", "price"]].corr()
```
![image](https://github.com/user-attachments/assets/6b0e27f0-c7a3-4cae-8c8d-e68765c906af)
![image](https://github.com/user-attachments/assets/7ad5fc07-1fe3-4dd9-8160-185b4299123d)

Observations on the Relationship: The regression line has a steep slope, indicating that as engine size increases, car price also increases. Most data points are close to the regression line.
Some points are farther away (outliers), but the overall trend still suggests a strong linear relationship.
The correlation coefficient is 0.872335 high (close to 1) means there is a strong positive linear relationship between engine-size and the price

the (linear) relationship between an 'highway-mpg'  and the 'price'
```
sns.regplot(x='highway-mpg', y='price', data = df)
plt.ylim(0,)
plt.show()
df[["highway-mpg", "price"]].corr()
```
![image](https://github.com/user-attachments/assets/43baad50-f770-4107-8ac7-07f73e11f06f)
![image](https://github.com/user-attachments/assets/72aa5a1d-7637-49e0-bbe0-4a31b4d821af)

The chart and the correlation coefficient of -0.704692 indicate a strong inverse relationship between highway fuel consumption (highway-mpg) and car price.
Real-world prediction: If a car has a high mpg, we can expect its price to be lower.

the (linear) relationship between an 'Peak RPM'  and the 'price'
```
sns.regplot(x='peak-rpm', y='price', data = df)
plt.ylim(0,)
plt.show()
df[["hpeak-rpm", "price"]].corr()
```
![image](https://github.com/user-attachments/assets/69b1872a-1c4a-43d5-a5bd-6407e1827ffd)
![image](https://github.com/user-attachments/assets/77937c36-1adb-478d-97eb-5edfdc9fdc57)

Peak RPM is not a significant factor in determining car prices.
Other factors such as engine size, horsepower, brand, and fuel efficiency may have a greater impact on car prices.

the (linear) relationship between an 'Peak RPM'  and the 'price'
```

```
## Categorical Variables
relationship between "body-style" and "price"
```
sns.boxplot(x='body-style', y ='price', data = df)
plt.show()
```
![image](https://github.com/user-attachments/assets/2eff6cf9-dff3-4967-a7b8-b58520d00590)

Based on the boxplot, predicting price solely based on body style may not be highly accurate for several reasons:

High Variability – The price range for each body style is quite large. For example:
Convertible & Hardtop show the highest variation in prices.
Hatchbacks generally have lower prices but still exhibit some outliers.
Sedans & Wagons have a moderate spread.

Overlapping Price Ranges – Many body styles have overlapping price distributions. For instance, a high-end sedan might be more expensive than a cheap convertible, making it hard to distinguish prices just by body style.



Examine 
"engine-location" and "price"
```
sns.boxplot(x="engine-location", y="price", data=df)
plt.show()
```
![image](https://github.com/user-attachments/assets/70691f47-40dc-4313-9e19-8f4ea93920a3)

Rear-Engine Vehicles Are More Expensive – Cars with rear engine locations have significantly higher prices, with a much smaller price range. This suggests that rear-engine cars are typically luxury or high-performance vehicles.

Front-Engine Vehicles Have a Wide Price Range – The prices of front-engine vehicles vary greatly, from low-cost to high-end models. There are many outliers in the higher price range, indicating that some front-engine cars can be very expensive, but most are relatively affordable.


Examine  "drive-wheels" and "price"
```
sns.boxplot(x="drive-wheels", y="price", data=df)
plt.show()
```
![image](https://github.com/user-attachments/assets/5786faaa-5b6a-4c08-b338-f9c9bf37f8be)

RWD (Rear-Wheel Drive) Vehicles:
Have the highest price range and variability.
Many luxury or high-performance cars fall into this category, which contributes to higher prices.
The median price is significantly higher compared to fwd and 4wd cars.
Several outliers exceed $40,000.

FWD (Front-Wheel Drive) Vehicles:
Tend to be more affordable, with lower price variation.
The median price is lower than rwd cars.
A few outliers exist, but prices are mostly below $15,000.
Common in economy cars and sedans.

4WD (Four-Wheel Drive) Vehicles:
Generally have a higher base price than FWD, but lower than RWD on average.
Prices are more consistent with fewer outliers.
Some higher-end SUVs or off-road vehicles could be responsible for the higher prices.
Prediction Feasibility:
Drive-wheels influence price, but it's not the only factor.
RWD cars tend to be more expensive, while FWD cars are the most affordable.
Overlapping price distributions make it hard to predict exact prices solely based on drive-wheel type.
Other factors like brand, engine size, horsepower, and features also play a crucial role.

Conclusion:
Drive-wheels can be a useful feature in price prediction but should be combined with other variables for better accuracy. 

## Descriptive Statistical Analysis
The describe function automatically computes basic statistics for all continuous variables. Any NaN values are automatically skipped in these statistics.
```
df.describe()
```
![image](https://github.com/user-attachments/assets/304cd9cb-fbd9-4930-98a8-e3b883541376)

understanding how many units of each drive-wheels
```
df['drive-wheels'].value_counts().to_frame()
```
![image](https://github.com/user-attachments/assets/52dbe3f1-a76c-49be-96d2-33af21a62cba)

understanding how many units of each engine-location
```
df['engine-location'].value_counts().to_frame()
```
![image](https://github.com/user-attachments/assets/da8661e8-9b13-4622-956b-680267520339)

## Basics of Grouping
group by the variable "drive-wheels". We see that there are 3 different categories of drive wheels
```
df['drive-wheels'].unique() # check the type of 'drive-wheels'
```
![image](https://github.com/user-attachments/assets/8f8659d1-90cf-4185-b8f2-9960d1f09957)

check on average, which type of drive wheel is most valuable, we can group "drive-wheels" and then average them
```
df_1 = df[['drive-wheels', 'body-style', 'price']] # select columns need to be grouped.
group_1 = df_1.groupby(['drive-wheels', 'body-style'], as_index=False).agg({'price': 'mean'})
group_2 = df_1.groupby(['drive-wheels'], as_index=False).agg({'price': 'mean'})
group_1 
group_2
```
![image](https://github.com/user-attachments/assets/d35ed1c7-2ee9-478a-a907-fa92d618fec4)

![image](https://github.com/user-attachments/assets/4be88265-346f-40d1-83ba-c95f3751161f)

From data, it seems rear-wheel drive vehicles are, on average, the most expensive, while 4-wheel and front-wheel are approximately the same in price.

leave the drive-wheels variable as the rows of the table, and pivot body-style to become the columns of the table
```
group_1_pivot = group_1.pivot(index='drive-wheels', columns = 'body-style').fillna(0)
group_1_pivot['Row Average'] = group_1_pivot.mean(axis=1)
group_1_pivot.loc['Col Average'] = group_1_pivot.mean(axis=0)
group_1_pivot
```
![image](https://github.com/user-attachments/assets/92a86460-9c25-4a30-8d21-1444ace8f51a)

check on average, which type of body-style is most valuable, we can group "body-style" and then average them
```
group_3= df_1.groupby(['body-style'], as_index=False).agg({'price': 'mean'})
group_3
```
From data, it seems hardtop and convertible vehicles are, on average, the most expensive, while sedan and sedan are approximately the same in price in the middle, hatchback is cheapest price.

visualize the relationship between Body Style, drive-wheels vs Price
```
plt.pcolor(group_1_pivot, cmap='RdBu')
plt.xticks(np.arange(0.5, len(group_1_pivot.columns), 1), group_1_pivot.columns.levels[1], rotation=90)
plt.yticks(np.arange(0.5, len(group_1_pivot.index), 1), group_1_pivot.index)
plt.colorbar()
plt.show()
```
![image](https://github.com/user-attachments/assets/684b0058-c558-4045-8a30-6f8c8f550fec)

Luxury cars → RWD drive type (convertibles & sedans) are the most expensive.
Budget cars → FWD drive type (hatchbacks & wagons) are the cheapest.
4WD cars have a mix of low and mid-range pricing, depending on their use case.

## Correlation and Causation
```
from scipy import stats # import scipy package
```
p-value is 0.001: we say there is strong evidence that the correlation is significant.

the p-value is 0.05: there is moderate evidence that the correlation is significant.

the p-value is 0.1: there is weak evidence that the correlation is significant.

the p-value is 0.1: there is no evidence that the correlation is significant.



calculate the Pearson Correlation Coefficient and P-value of numberic columns and 'price'.
```
columns_to_check = ['wheel-base', 'horsepower', 'length', 'width', 'curb-weight', 'engine-size','bore', 'city-mpg', 'highway-mpg']
for column in columns_to_check :
    pearson_coef, p_value = stats.pearsonr(df[column], df['price'])
    print(f"The Pearson Correlation Coefficient between {column} is", pearson_coef, " with a P-value of P =", p_value)
```
![image](https://github.com/user-attachments/assets/3b71cc97-99c0-470a-9bcf-046d74e143e2)

The Pearson correlation analysis shows that engine size (0.872), curb weight (0.834), and horsepower (0.810) have the strongest positive correlations with price, indicating that as these values increase, price tends to increase as well. Conversely, city-mpg (-0.687) and highway-mpg (-0.705) show strong negative correlations, meaning that better fuel efficiency is generally associated with lower car prices. All correlations have extremely low p-values, confirming that these relationships are statistically significant.
