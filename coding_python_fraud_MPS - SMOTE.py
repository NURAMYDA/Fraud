# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 07:01:42 2020

@author: Nurhamidah Farhana
"""

import os
path = os.getcwd()
print(path)

#IMPORTING LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import f1_score, recall_score
from yellowbrick.classifier import ClassificationReport
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def PrintStats(cmat, y_test, pred):
   # separate out the confusion matrix components
   tpos = cmat[0][0]
   tneg = cmat[1][1]
   fpos = cmat[0][1]
   fneg = cmat[1][0]
   # calculate F!, Recall scores
   f1Score = round(f1_score(y_test, pred), 3)
   recallScore = round(recall_score(y_test, pred), 3)
   # calculate and display metrics
   print(cmat)
   print( 'Accuracy: '+ str(np.round(100*float(tpos+tneg)/float(tpos+fneg + fpos + tneg),2))+'%')
   print( 'Cohen Kappa: '+ str(np.round(cohen_kappa_score(y_test, pred),3)))
   print("Sensitivity/Recall for Model : {recall_score}".format(recall_score = recallScore))
   print("F1 Score for Model : {f1_score}".format(f1_score = f1Score))

def RunModel(model, X_train, y_train, X_test, y_test):
   model.fit(X_train, y_train.values.ravel())
   pred = model.predict(X_test)
   matrix = confusion_matrix(y_test, pred)
   return matrix, pred


def plot_confusion_matrix(cmat, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cmat = cmat.astype('float') / cmat.sum(axis=1)[:, np.newaxis]
        print("Confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cmat)


    fig, ax = plt.subplots()
    im = ax.imshow(cmat, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cmat.shape[1]),
           yticks=np.arange(cmat.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cmat.max() / 2.
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(j, i, format(cmat[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cmat[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#IMPORTING DATASET
df = pd.read_csv('E:\MASTER\Project P1\Dataset\python\paysim.csv')
df.info()
## There are 6362620 observations and 11 attributes


#HANDLING MISSING DATA
#Identify if there is any null value (missing data)
df.isnull().values.any()
##'False' indicate that there is none null value (empty observation)

#Rename the attributes
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrig', 'newbalanceOrig':'newBalanceOrig', 
                        'oldbalanceDest':'oldBalanceDest', 'newbalanceDest':'newBalanceDest'})

#to give a statistical summary of the dataset
a = df.describe(include='all')
print(a)

#identify on which payment types that have fraudulent transactions
df1 = df[df.isFraud == 1]
b = df1.describe(include='all')
print(b)
print(df1.type.value_counts())
print(df1.nameOrig.value_counts())
print(df1.nameDest.value_counts())
#based on the fraudulent data, only payment type of CASH_OUT and TRANSFER that have fraudulent transactions, hence we will only select the dataset for payment type CASH_OUT and TRANSFER for further analysis
options = ['CASH_OUT', 'TRANSFER']
df2 = df[df['type'].isin(options)]

#try dframe = df2.concat(df1,df2)
c = df2.describe(include='all')
print(c)

#define class for target variable
class_names = {0:'Not Fraud', 1:'Fraud'}
print(df2.isFraud.value_counts().rename(index = class_names))


#HANDLING CATEGORICAL DATA
#import the necessary module to change from string labels to numeric labels 
from sklearn import preprocessing
# create the Labelencoder object
le = preprocessing.LabelEncoder()
#convert the categorical columns into numeric
encoded_value = le.fit_transform(["TRANSFER", "CASH_OUT"])
print(encoded_value)

#convert the categorical columns into numeric
df2['type'] = le.fit_transform(df2['type'])
df2['nameOrig'] = le.fit_transform(df2['nameOrig'])
df2['nameDest'] = le.fit_transform(df2['nameDest'])
#display the initial records
df2.head()

#correlation test
corr = df2.corr()
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df2.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df2.columns)
ax.set_yticklabels(df2.columns)
plt.show()


#since variable isFlaggedFraud is less significant, it will be remove and a new dataset will be created
df_new = df2.drop(columns=['isFlaggedFraud'])
print(df_new.head)

#Splitting training and test set
feature_names = df_new.iloc[:, 0:9].columns
target = df_new.iloc[:1, 9: ].columns

data_features = df_new[feature_names]
data_target = df_new[target]
data_features.info()
data_target.info()
#use splitting ratio of 50:50, 60:40, 70:30, 80:20, 90:10
from sklearn.model_selection import train_test_split
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(data_features, data_target, train_size=0.70, test_size=0.30, random_state=1)


print(sorted(Counter(y_train['isFraud']).items()))


#SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=1)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(sorted(Counter(y_train['isFraud']).items()))

#Feature scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#MACHINE LEARNING
##LogisticRegression
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
cmat, pred = RunModel(lr, X_train, y_train, X_test, y_test)
PrintStats(cmat, y_test, pred)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('LR_cm.png')
# Plot normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('LR_cm_normal.png')

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(lr, classes=['0','1'])
visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
g = visualizer.poof()


#KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
cmat, pred = RunModel(neigh, X_train, y_train, X_test, y_test)
PrintStats(cmat, y_test, pred)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('KNN_cm.png')
# Plot normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('KNN_cm_normal.png')

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(neigh, classes=['0','1'])
visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data


#Decision tree
from sklearn import tree
dt = tree.DecisionTreeClassifier()
cmat, pred = RunModel(dt, X_train, y_train, X_test, y_test)
PrintStats(cmat, y_test, pred)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('DT_cm.png')
# Plot normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('DT_cm_normal.png')

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(dt, classes=['0','1'])
visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data


##RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, n_jobs =4)
cmat, pred = RunModel(rf, X_train, y_train, X_test, y_test)
PrintStats(cmat, y_test, pred)

# Plot non-normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig('RF_cm.png')
# Plot normalized confusion matrix
plot_confusion_matrix(cmat, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.savefig('RF_cm_normal.png')

# Instantiate the classification model and visualizer
visualizer = ClassificationReport(rf, classes=['0','1'])
visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
visualizer.score(X_test, y_test) # Evaluate the model on the test data
g = visualizer.poof() # Draw/show/poof the data
