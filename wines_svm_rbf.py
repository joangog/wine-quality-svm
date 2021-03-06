import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# parameters
kernel = 'rbf'
C = [0.001,0.01,0.1,1,10,100]
GAMMA = [0.001,0.01,0.1,1,10,100]

# vars
f1 = np.zeros((len(C),len(GAMMA))) # f1 for each parametrisation
precision = np.zeros((len(C),len(GAMMA))) # precision for each parametrisation
recall = np.zeros((len(C),len(GAMMA))) # recall for each parametrisation

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

# split dataset
train_dataset = dataset.iloc[:int(0.75*len(dataset))]
test_dataset = dataset.loc[int(0.75*len(dataset)):]

# train support vector classifier
x_train = train_dataset.iloc[:,:-1] # train inputs
y_train = train_dataset.iloc[:,-1] # train outputs
labels = list(range(0,11)) # class labels (we assume possible ratings are 1 to 10 so 11 possible classes)
x_test = test_dataset.iloc[:,:-1] # test inputs
y_test_true = test_dataset.iloc[:,-1] # true test outputs
for i,c in enumerate(C): # for each value of c
    for j,gamma in enumerate(GAMMA): # for each value of gamma
        svc = SVC(C=c,gamma=gamma,kernel=kernel)
        svc.fit(x_train,y_train) # train svc
        y_test_pred = svc.predict(x_test) # predicted test outputs

        #calculate metrics for specific parameters
        # Note: average=macro means that the metrics are calculated as the unweighted average for every class
        #       zero_division = 1 because it raised a warning when all samples where classified as positive or negative for a specific class
        f1[i,j] = metrics.f1_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1)
        precision[i,j] = metrics.precision_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1)
        recall[i,j] = metrics.recall_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1)

# find the parameters that give the best metrics
max_f1_indices = np.unravel_index(np.argmax(f1, axis=None), f1.shape) # argmax returns the flat index of max value so we use unravel index to convert it to a tuple of indices
max_f1_pars = (C[max_f1_indices[0]],GAMMA[max_f1_indices[1]])
max_precision_indices = np.unravel_index(np.argmax(precision, axis=None), precision.shape)
max_precision_pars = (C[max_precision_indices[0]],GAMMA[max_precision_indices[1]])
max_recall_indices = np.unravel_index(np.argmax(recall, axis=None), recall.shape)
max_recall_pars = (C[max_recall_indices[0]],GAMMA[max_recall_indices[0]])

# print results
print("Best results:")
print(f'- F1 = {f1.max():.4f} for (c,gamma) = {max_f1_pars}')
print(f'- Precision = {precision.max():.4f} for (c,gamma) = {max_precision_pars}')
print(f'- Recall = {recall.max():.4f} for (c,gamma) = {max_recall_pars}')