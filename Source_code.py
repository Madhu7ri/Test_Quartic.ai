import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_predict, cross_val_score, KFold
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
import warnings
from nltk.chunk.util import accuracy
from unicodedata import decimal
warnings.filterwarnings("ignore")

print("LOADING THE DATA....")
df_train = pd.read_csv("/home/madhuri/Desktop/ds_data/data_train1.csv",header = 0)
y_train = pd.read_csv("/home/madhuri/Desktop/ds_data/x_target.csv",header = 0)

#print(df_train.isnull().sum())

df_test = pd.read_csv("/home/madhuri/Desktop/ds_data/data_test.csv",header = 0)
#print(df_test.isnull().sum())

print("Clean and prepare data....")
df_test.fillna(df_test.median(), inplace=True)

# For a look of missing values in test data
#print(df_test.isnull().sum()) 

df_train.fillna(df_train.median(), inplace=True)
#For a look for missing values in training data
#print(df_train.isnull().sum())

print("Data Looks Like..")
print(y_train.head())
#print(np.count_nonzero(y_train))# to count number of 1 in target to estimate the class distribution

#Drop First column
df_train.drop('id',axis=1,inplace=True)
df_test.drop('id',axis=1,inplace=True)

print("View of dataset....")
print("Size of training data", df_train.shape) 
print("Size of target data", y_train.shape)
print("Size of test data",df_test.shape) 


### Training ##############################################################################

#Generic function for making a classification model and accessing the performance. 
print("TRAINING.....")
def classification_model(clf, train_x, train_y, test_x):
    tree=clf.fit(train_x,train_y) #Fit the model
    y_pred = clf.predict(test_x) ##Make predictions on training set:
    fv=cross_val_score(tree,test_x,y_pred,cv=10).mean()
    print "10_KfoldCrossVal score using model is:", "%.2f" %fv
    #return Prediction
    return y_pred

### Models########################################################################### 


#clf=LogisticRegression()
clf=GaussianNB()

T=classification_model(clf,df_train, y_train, df_test)
print("Target is", T)
#Save result in csv file
np.savetxt('/home/madhuri/Desktop/target1.csv',T, delimiter=',')
print("Done..Output saved in CSV file")



