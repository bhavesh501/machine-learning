import numpy as np
np.set_printoptions(threshold = np.nan)
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
y[:, 3] = imputer.transform(y[:, 3])
data = x

#displaying features
list(dataset)


#encoding the data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

"""#feature extraction
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
pos = vec.fit_transform(x).toarray()
vec.get_feature_names()"""

#using cross-validation
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state=0)

#using linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predicting the results
y_pred = regressor.predict(x_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)