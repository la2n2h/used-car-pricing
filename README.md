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

# MODEL DEVELOPMENT
develop several models that will predict the price of the car using the variables or features. This is just an estimate but should give us an objective idea of how much the car should cost.

import package
```
! pip install seaborn
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seabon as sns
```
import data 
```
df = pd.read_csv('D:/Data anlysis- working sheet/python/data/auto_clean_df.csv')
df.head()
```

## Linear Regression and Multiple Linear Regression
Simple Linear Regression is a method to help us understand the relationship between two variables:

The predictor/independent variable (X)
The response/dependent variable (that we want to predict)(Y)
The result of Linear Regression is a linear function that predicts the response (dependent) variable as a function of the predictor (independent) variable.

```
from sklearn.linear_model import LinearRegression # load the modules for linear regression
```

initialize a linear regression model, but it has not been trained on any data yet
```
lm = LinearRegression()
lm
```
![image](https://github.com/user-attachments/assets/a8f7997f-2bea-4d6f-888c-a5acb4d5a0d6)

Using simple linear regression, we will create a linear function with "highway-mpg" as the predictor variable and the "price" as the response variable.
```
x = df[['highway-mpg']] # predictor variable
y = df[['price']] # response variable
lm.fit(x, y) #Fit the linear model using highway-mpg
Yhat=lm.predict(x) #output a prediction
Yhat[0:5]
```
![image](https://github.com/user-attachments/assets/21712d25-d77e-4aef-b9ce-0480f3b595fd)

value of the intercept (a)
```
lm.intercept_
```
![image](https://github.com/user-attachments/assets/52843b62-b3c7-4593-96cb-386ce7c1a8ea)

value of slope (b)
```
lm.coef_
```
![image](https://github.com/user-attachments/assets/a128b682-55f0-499b-a4d1-2c1ba8ee8e77)

get a final linear model with the structure: Yhat = a + bX

Price = 38423.31 - 821.73 x highway-mpg

plot a linear regression graph
```
sns.regplot (x = x, y = y)
plt.xlabel("Highway MPG")
plt.ylabel("Price")
plt. title("Linear Regression: Price vs Highway MPG")
plt.ylim(0,)
plt.show()
```
![image](https://github.com/user-attachments/assets/04a03444-dbca-41eb-b4d1-4095edf40bcf)

Negative correlation:
The regression line has a negative slope → As Highway MPG increases, Price decreases.
This makes sense: Cars with higher fuel efficiency tend to be cheaper.

Model fit:
The data points do not perfectly align with the regression line, showing significant scatter.
This suggests that the relationship between Price and Highway MPG is not strictly linear.

Wide light blue confidence band: This suggests high uncertainty in the model, meaning Highway MPG is not the sole factor affecting car prices.
Presence of outliers: Some cars have very high or very low prices that do not follow the general trend.
```
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()
```
![image](https://github.com/user-attachments/assets/bed819ad-f5f4-4db2-81c1-fb779302f94b)

X-axis: highway-mpg.
Y-axis: residual (the error between the actual value and the prediction).
Dotted line: The zero mark, meaning that if the residual is evenly distributed around this line, the model is suitable.
📌 Observe the chart:

The residual is not randomly distributed around 0.
At both ends (low & high highway-mpg), the residual tends to increase or decrease sharply.
A curvilinear pattern (U-shape) appears.
⏳ ⛔ Conclusion:

This is a sign that the linear model is not suitable. Let's see if we can try fitting a polynomial model to the data instead.

```
# creaete function to plot the data
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# get the variables
x = df['highway-mpg']
y = df['price']


#fit the polynomial using the function polyfit, then use the function poly1d to display the polynomial function.
f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)


# Let's plot the function
PlotPolly(p, x, y, 'highway-mpg')
```
![image](https://github.com/user-attachments/assets/12cf6969-d57a-4e72-a3b6-d9aec368bd96)


haracteristics of the Chart:
✅ Original Data (blue dots): Represents the actual data points.
✅ Orange Curve: The fitted polynomial regression model of degree 3.
✅ Polynomial Equation Displayed on the Chart:

 
This equation is used to predict car prices based on miles per gallon.

Observations:
The data is nonlinear, meaning a simple linear regression (straight line) is not suitable.
A third-degree polynomial model fits better since it captures the nonlinear variations in the data.
Trend: As highway-mpg increases (the car becomes more fuel-efficient), car price decreases.
Saturation Effect: When highway-mpg is very high (above 40), car prices do not drop significantly anymore.

Conclusion:
Polynomial regression is a better choice when data has a nonlinear relationship.
This model can predict car prices more accurately than linear regression.

## Multiple Linear Regression
explain the relationship between one continuous response (dependent) variable and two or more predictor (independent) variables
```
lm2 = LinearRegression()
x1 = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
y1 = df[['price']]
lm2. fit(x1, y1)
lm2.intercept_
lm2.coef_
```
a = array([-15811.86376773])
b = array([[53.53022809,  4.70805253, 81.51280006, 36.1593925 ]])

Yhat = -15811.863 + 53.53 *'horsepower' + 4.70* 'curb-weight' + 81.51*'engine-size' + 36.159* 'highway-mpg'

look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.
```
Y_hat = lm2.predict(x1)
ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)
```
![image](https://github.com/user-attachments/assets/b3d4bcaa-d59f-4d83-ae17-05731397ab31)

Red Line: Likely represents the actual price distribution
Blue Line: Likely represents the predicted price distribution from a regression model

Peak of the chart

Both curves have peaks around 10,000 - 15,000, indicating that most car prices fall within this range.
The red curve is slightly higher, suggesting that actual prices are more concentrated in this range.
Tail section (20,000 and above)

Higher price values (>20,000): There are some differences between the two curves.
The red curve has a small bump around 30,000 - 40,000, indicating a subset of cars with higher prices.
Differences between actual and predicted values

If the blue curve represents predicted prices from the model, the model performs well in the common price range (10,000 - 20,000).
However, it may struggle to predict higher prices (30,000+).

### We can perform a polynomial transform on multiple features. 
we import the module
```
from sklearn.preprocessing import PolynomialFeatures
```
Create a PolynomialFeatures object of degree 2
```
pr=PolynomialFeatures(degree=2)
pr
```

```
x1_pr=pr.fit_transform(x1)
```

``
x1.shape
```
In the original data, there are 201 samples and 4 features.
```
```
Z_pr.shape
```
After the transformation, there are 201 samples and 15 features.

### Pipeline
Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.
```
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```
create the pipeline by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
```
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
```
input the list as an argument to the pipeline constructor
```
pipe=Pipeline(Input)
pipe
```
![image](https://github.com/user-attachments/assets/d0ea24df-bd2e-4bb7-b951-60a79421e43c)

First, we convert the data type x1 to type float to avoid conversion warnings that may appear as a result of StandardScaler taking float inputs.
Then, we can normalize the data, perform a transform and fit the model simultaneously.
```
Z = x1.astype(float)
pipe.fit(x1,y)
```
![image](https://github.com/user-attachments/assets/44643580-971c-4977-a72f-0c0655eb2872)

Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously.
```
ypipe=pipe.predict(Z)
ypipe[0:4]
```
![image](https://github.com/user-attachments/assets/cb91305c-6ff7-489c-87cb-d11ded7ae4e6)

Create a pipeline that standardizes the data, then produce a prediction using a linear regression model using the features Z and target y.
```
Input=[('scale',StandardScaler()),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
ypipe=pipe.predict(Z)
ypipe[0:10]
```
###  Measures for In-Sample Evaluation
When evaluating our models, not only do we want to visualize the results, but we also want a quantitative measure to determine how accurate the model is.
Model 1: Simple Linear Regression
```
lm.fit(x, y)
# Find the R^2
print('The R-square is: ', lm.score(x, y))
```
![image](https://github.com/user-attachments/assets/6a4ec737-5f14-4454-a443-55a7a8d01148)

 predict the output i.e., "yhat" using the predict method, where X is the input variable. calculate the MSE
```
Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
```
The output of the first four predicted value is:  [16236.50464347 16236.50464347 17058.23802179 13771.3045085 ]

import the function mean_squared_error from the module metrics:
```
from sklearn.metrics import mean_squared_error
```
```
compare the predicted results with the actual results
```
Model 2: Multiple Linear Regression
calculate the R^2:
```
lm.fit(Z, df['price'])
# Find the R^2
print('The R-square is: ', lm.score(Z, df['price']))
```
We can say that ~80.896 % of the variation of price is explained by this multiple linear regression "multi_fit".

calculate the MSE
```
Y_predict_multifit = lm.predict(Z)
```
compare the predicted results with the actual results
```
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))
```

Model 3: Polynomial Fit
calculate the R^2
```
from sklearn.metrics import r2_score
```
apply the function to get the value of R^2
```
r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)
```
calculate the MSE
```
mean_squared_error(df['price'], p(x))
```

## Prediction and Decision Making
```
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline 
```
Create a new input
```
new_input=np.arange(1, 100, 1).reshape(-1, 1)
```
Fit the model
```
lm.fit(X, Y)
lm
```
Produce a prediction
```
yhat=lm.predict(new_input)
yhat[0:5]
```
plot the data
```
plt.plot(new_input, yhat)
plt.show()
```

![image](https://github.com/user-attachments/assets/53089737-fd64-4518-bdd3-a420d5ae5fc4)

## Decision Making: Determining a Good Model Fit

Now that we have visualized the different models, and generated the R-squared and MSE values for the fits, how do we determine a good model fit?

What is a good R-squared value?
When comparing models, the model with the higher R-squared value is a better fit for the data.

What is a good MSE?
When comparing models, the model with the smallest MSE value is a better fit for the data.

Let's take a look at the values for the different models.

Simple Linear Regression: Using Highway-mpg as a Predictor Variable of Price.

R-squared: 0.49659118843391759
MSE: 3.16 x10^7

Multiple Linear Regression: Using Horsepower, Curb-weight, Engine-size, and Highway-mpg as Predictor Variables of Price.

R-squared: 0.80896354913783497
MSE: 1.2 x10^7

Polynomial Fit: Using Highway-mpg as a Predictor Variable of Price.

R-squared: 0.6741946663906514
MSE: 2.05 x 10^7

Simple Linear Regression Model (SLR) vs Multiple Linear Regression Model (MLR)
MSE: We can see that Polynomial Fit brought down the MSE, since this MSE is smaller than the one from the SLR.
R-squared: The R-squared for the Polynomial Fit is larger than the R-squared for the SLR, so the Polynomial Fit also brought up the R-squared quite a bit.
Since the Polynomial Fit resulted in a lower MSE and a higher R-squared, we can conclude that this was a better fit model than the simple linear regression for predicting "price" with "highway-mpg" as a predictor variable.

Multiple Linear Regression (MLR) vs. Polynomial Fit
MSE: The MSE for the MLR is smaller than the MSE for the Polynomial Fit.
R-squared: The R-squared for the MLR is also much larger than for the Polynomial Fit.
Conclusion
Comparing these three models, we conclude that the MLR model is the best model to be able to predict price from our dataset. This result makes sense since we have 27 variables in total and we know that more than one of those variables are potential predictors of the final car price.

# MODEL EVALUATION AND REFINEMENT

### Functions for Plotting
```
# create a function to build a Distribution chart
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figzize = (width, height))

    # draw a Kernel Density Estimation (KDE)
    ax1 = sns.kdeplot(RedFunction, color = 'r', label = RedName)
    ax2 = sns.kdeplot(BlueFunction, color = 'b', label = BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plot.show()
    plot.close()

# create a function to build a Polynomial Regression plot
def PollyPlot(xtrain, xtest, y_train, y_text, lr, poly_transform):
    width = 12
    height = 10
    plot.figure(figsize = (width, height))

# training data
# testing data
# lr: liner regression object
    xmax = max(xtrain.values.max(), xtest.values.max())
    xmin = min(xtrain.values.min(), xtest.values.min())

    x = np.arrange(xmin, xmax, 0.1)

    plt.plot(xtrain, ytrain, 'ro', label = "Training data")
    plt.plot(x_test, y_test, 'go', label = 'Testing data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label = 'Predicted Function')
    plt.ylim([-10000, 60000])
    pl.ylabel('Price')
    plt.legend()
```
### Training and Testing
```
# place the target data price in a separate dataframe y_data
y_data = df['price']

# Drop price data in dataframe x_data
x_data=df.drop('price',axis=1)

# randomly split our data into training and testing data using the function train_test_split
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

print('number of test samples:', x_test.shape[0])
print('number of training samples:', x_train.shape[0])
```
![image](https://github.com/user-attachments/assets/c356e9e4-398d-4735-aa9f-759ee024aba0)


