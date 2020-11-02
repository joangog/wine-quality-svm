import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# parameters
kernel = 'linear' # svc
c = 10 # svc
k = 50 # kmeans

# vars
f1 = []  # f1 for each loop
precision = []  # precision for each loop
recall = []  # recall for each loop

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

# fill missing values with average value of column to prepare for kmeans
avg_ph = np.nanmean(dataset['pH'])
for i in range(int(len(dataset))):
    if np.isnan(dataset.at[i,'pH']):
        dataset.at[i,'pH'] = avg_ph

#kmeans for 50 loops
for i in range(0,50):
    # train kmeans model to fill missing values
    kmeans = KMeans(k,init='random') # initial centroids are random
    clusters = kmeans.fit_predict(dataset)
    for i in range(k): # for every cluster
        avg_cluster_ph = np.mean(dataset.iloc[np.where(clusters==i)]['pH'])
        for j in range(int(len(dataset)*0.33)): # for every wine with missing value
            if clusters[j] == i: # if the wine belongs to the current cluster
                dataset.at[j,'pH'] = avg_cluster_ph  # replace the avg value with the new average cluster ph value

    # split dataset
    train_dataset = dataset.iloc[:int(0.75*len(dataset))]
    test_dataset = dataset.loc[int(0.75*len(dataset)):]

    # train support vector classifier to predict wine quality
    x_train = train_dataset.iloc[:,:-1] # train inputs
    y_train = train_dataset.iloc[:,-1] # train outputs
    labels = list(range(0,11)) # class labels (we assume possible ratings are 1 to 10 so 11 possible classes)
    x_test = test_dataset.iloc[:,:-1] # test inputs
    y_test_true = test_dataset.iloc[:,-1] # true test outputs
    svc = SVC(C=c,kernel=kernel)
    svc.fit(x_train,y_train) # train svc
    y_test_pred = svc.predict(x_test) # predicted test outputs

    # calculate metrics for loop
    # Note: average=macro means that the metrics are calculated as the unweighted average for every class
    #       zero_division = 1 because it raised a warning when all samples where classified as positive or negative for a specific class
    f1.append(metrics.f1_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1))
    precision.append(metrics.precision_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1))
    recall.append(metrics.recall_score(y_test_true,y_test_pred,labels,average='macro',zero_division=1))

# calculate avg metrics
avg_f1 = np.mean(f1)
avg_precision= np.mean(precision)
avg_recall = np.mean(recall)

# print results
print(f'Results for {k}-Means:')
print(f'- F1: = {avg_f1:.4f}')
print(f'- Precision = {avg_precision:.4f}')
print(f'- Recall = {avg_recall:.4f}')