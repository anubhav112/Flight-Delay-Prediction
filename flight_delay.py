import numpy as np
import pandas as pd
from sklearn import preprocessing,model_selection,svm,neighbors,linear_model

df = pd.read_csv("/home/anubhav/Downloads/flight_delay_prediction/dataset.csv")
df.drop(['YEAR',
	'MONTH',
	'DAY_OF_MONTH',
	'CARRIER',
	'ORIGIN_AIRPORT_ID',
	'DEST_AIRPORT_ID',
	'CRS_ARR_TIME',
	'ARR_DELAY',
	'ARR_DEL15',
	'SkyCondition'],1,inplace = True)

for i in range(len(df['CANCELLED'])):
	if(df['CANCELLED'][i] == 1):
		df.drop(i,0,inplace = True)
df.drop(['CANCELLED'],axis = 1,inplace = True)

df['DryBulbCelsius'] = pd.to_numeric(df['DryBulbCelsius'],errors = 'coerce')
df['Visibility'] = pd.to_numeric(df['Visibility'],errors = 'coerce')
df['WindSpeed'] = pd.to_numeric(df['WindSpeed'],errors = 'coerce')
df['WindDirection'] = pd.to_numeric(df['WindDirection'],errors = 'coerce')
df['StationPressure'] = pd.to_numeric(df['StationPressure'],errors = 'coerce')

df.fillna(df.mean(),inplace = True)
# print(df.describe())
# print(df.mean())
# for i in df:
# 	print(df[i].count())

def create_classes(y):
	for i in range(len(y)):
		y[i] = int(y[i]/15.0)
		# print(y[i])
	return y

df.astype('float64')
y = np.array(df['DEP_DELAY_NEW'])
y = create_classes(y)
X = np.array(df.drop(['DEP_DELAY_NEW','DEP_DEL15'],axis = 1))
# X = preprocessing.scale(X)

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

input_dict = ['DAY_OF_WEEK','FL_NUM','CRS_DEP_TIME','DryBulbCelsius','Visibility','WindSpeed','WindDirection','StationPressure']

test = []
for i in input_dict:
	print(i,"for the plane is : ",end = " ")
	test.append(np.float(input()))
test = np.array(test)
test = np.reshape(test,(-1,test.shape[0]))
output = clf.predict(test)
print(output)
