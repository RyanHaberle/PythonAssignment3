import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import seaborn as sns
from sklearn import metrics

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.head())
sns.pairplot(train, x_vars=['LotArea','LotShape','GrLivArea','FullBath', 'GrLivArea','WoodDeckSF' ,'LotFrontage' ,'BsmtHalfBath','MSSubClass' ], y_vars="SalePrice", height=7, aspect=0.7)

plt.scatter(x=train['MSSubClass'],y=np.log(train.SalePrice))
plt.xlim(-200,1600)
plt.ylabel('Sale Price')
plt.xlabel('MSSubClass')
plt.show()

plt.scatter(x=train['1stFlrSF'],y=np.log(train.SalePrice))
plt.xlim(-200,1600)
plt.ylabel('Sale Price')
plt.xlabel('1stFlrSF')
plt.show()

data = train.select_dtypes(include='number').interpolate().dropna()

X = data.drop(['SalePrice','Id','MSSubClass','MoSold','YrSold','PoolArea','BsmtHalfBath','MSSubClass'], axis=1)
y = np.log(train.SalePrice)

linReg = linear_model.LinearRegression()




X_train, X_test,y_train,y_test = train_test_split(X,y)

model = linReg.fit(X_train,y_train)

features = test.select_dtypes(include=[np.number]).drop(['Id','MSSubClass','MoSold','YrSold','PoolArea','BsmtHalfBath'], axis=1).interpolate()
print(features)
predictions = model.predict(features)

y_pred = linReg.predict(X_test)
y_actual = y_test
plt.scatter(y_pred, y_test)
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')

plt.show()
print(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
print(mean_squared_error(y_test,y_pred))

submissionFile = pd.DataFrame()

submissionFile['id'] = test.Id
submissionFile['SalePrice'] = np.exp(predictions)
submissionFile.to_csv('kagglefile.csv')
print(y_test.shape)
print(submissionFile.shape)


print(submissionFile)

