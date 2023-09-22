import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
#columns = ['Days Lost','Incident Cost']
Safety = pd.read_excel('sampledatasafety.xlsx',sheet_name='SafetyData')

#sns.pairplot(Safety,x_vars=['Days Lost'],y_vars=['Incident Cost'],hue='Gender')
#plt.show()

#print(Safety.corr())

X = Safety[['Age Group', 'Incident Type', 'Gender',]]
X = pd.get_dummies(data=X, drop_first=True)

Y = Safety['Report Type']
Y = pd.get_dummies(data=Y, drop_first=True)



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=10, shuffle=True)

model = LinearRegression().fit(X_train, y_train)


y_prediction = model.predict(X_train)
print("MAE on train data= " , metrics.mean_absolute_error(y_train, y_prediction))

y_prediction = model.predict(X_test)
print("MAE on test data = " , metrics.mean_absolute_error(y_test, y_prediction))
