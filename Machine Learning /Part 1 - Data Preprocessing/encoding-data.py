#encoding the categorical data
#encoding independent variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()
#encoding dependent variables
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)