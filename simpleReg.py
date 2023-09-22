import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Defining the columns of and reading our DataFrame
columns = ['Age','Premium']
Data = pd.read_csv('simplelinearregression.csv')

# Printing the head of our DataFrame
#print(Data.head())
#print(Data.corr())

x = Data[['Age']]
y = Data['Premium']

#Z = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=10, shuffle=True)

model = LinearRegression().fit(X_train, y_train)

y_prediction = model.predict(X_train)
print("MAE on train data= " , metrics.mean_absolute_error(y_train, y_prediction))
acc=model.score(X_train, y_train)
print("Accuracy: "+ str(acc))
# Evaluating the trained model on test data
y_prediction = model.predict(X_test)
print("MAE on test data = " , metrics.mean_absolute_error(y_test, y_prediction))
acc=model.score(X_test, y_test)
print("Accuracy: "+ str(acc))

#plt.plot(x, y,'o')
#plt.show()
