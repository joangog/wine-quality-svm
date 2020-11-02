import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# parameters
kernel = 'linear'
c = 10

# create dataset
dataset = pd.read_csv('winequality-red.csv')
features = dataset.columns # feature names

# preprocessing
sc = StandardScaler()
x = dataset.iloc[:,:-1] # take the input columns
x = pd.DataFrame(sc.fit_transform(x)) # preprocess input data
y = dataset.iloc[:,-1] # take the output column
dataset = pd.concat([x,y],1) # recreate dataset from x and y
dataset.columns = features # add feature names

# remove 33% values from pH column
for i in range(int(len(dataset)*0.33)):
    dataset.at[i,'pH'] = None

# train logistic regression model using existing values
linreg = LinearRegression()
ph_test_dataset = dataset.iloc[:int(len(dataset)*0.33)] # test data will be those where pH is missing
ph_train_dataset = dataset.iloc[int(len(dataset)*0.33):] # train data will be those where pH is known
linreg.fit(ph_train_dataset.drop(columns='pH'),ph_train_dataset['pH']) # train using as input all columns except pH and output the pH column
ph_pred = linreg.predict(ph_test_dataset.drop(columns='pH'))

#fill missing values with predicted values
for i in range(int(len(dataset)*0.33)):
    dataset.at[i,'pH'] = ph_pred[i]

# split dataset
train_dataset = dataset.iloc[:int(0.75*len(dataset))]
test_dataset = dataset.loc[int(0.75*len(dataset)):]

# train support vector classifier
x_train = train_dataset.iloc[:,:-1] # train inputs
y_train = train_dataset.iloc[:,-1] # train outputs
labels = list(range(0,11)) # class labels (we assume possible ratings are 1 to 10 so 11 possible classes)
x_test = test_dataset.iloc[:,:-1] # test inputs
y_test_true = test_dataset.iloc[:,-1] # true test outputs
svc = SVC(C=c,kernel=kernel)
svc.fit(x_train,y_train) # train svc
y_test_pred = svc.predict(x_test) # predicted test outputs

# calculate metrics
# Note: average=macro means that the metrics are calculated as the unweighted average for every class
#       zero_division = 1 because it raised a warning when all samples where classified as positive or negative for a specific class
f1 = metrics.f1_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1)
precision =  metrics.precision_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1)
recall = metrics.recall_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1)

# print results
print("Results:")
print(f'- F1 = {f1:.4f}')
print(f'- Precision = {precision:.4f}')
print(f'- Recall = {recall:.4f}')