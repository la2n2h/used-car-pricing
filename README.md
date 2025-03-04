# old-car

```
import pandas as pd
import numpy as np
```

```
file_path='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/auto.csv'
```

```
df = pd.read_csv(file_path)
df.head()
```
![image](https://github.com/user-attachments/assets/0cadf2af-6d54-4419-8fce-649cd0a46239)

```
df.tail(10)
```
![image](https://github.com/user-attachments/assets/e3270ff7-5c26-40d7-b224-60780add1a43)

```
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
```
```
df1 =df.replace('?', np.NaN)
df=df1.dropna(subset='price', axis=0)
df.head(20)
```
![image](https://github.com/user-attachments/assets/2c015df3-5f1f-41d9-a514-9dfac56849e0)

```
df.to_csv('D:/Data anlysis- working sheet/python/data/auto.csv', index=False)
```

```
df.describe(include = 'all')
```
![image](https://github.com/user-attachments/assets/1dcea16f-6571-467a-b0de-31abd41f02c3)

```
df.iloc[:,[10,20]].describe(include = 'all')
```
![image](https://github.com/user-attachments/assets/0ba06e7a-4614-426f-963c-79f8ebc60b8a)

```
df.info()
```
![image](https://github.com/user-attachments/assets/aa7051bf-8edb-4872-8fb1-d44b0f452e1a)
