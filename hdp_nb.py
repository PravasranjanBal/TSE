#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#### Load source data
s_data = pd.read_csv("F:\\HDP\\data\\Softlab\\ar6.csv")
#s_data = pd.read_csv("F:\\oodata_python\\ant\\ant-1.6.csv")


array = s_data.values
i1 = array[:,0]
i2 = array[:,1]
i3 = array[:,2]
i4 = array[:,3]
i5 = array[:,4]
i6 = array[:,5]
i7 = array[:,6]
i8 = array[:,7]
i9 = array[:,8]
i10 = array[:,9]
i11 = array[:,10]
i12 = array[:,12]
i13 = array[:,13]
i14 = array[:,14]
i15 = array[:,15]
i16 = array[:,16]
i17 = array[:,17]
i18 = array[:,18]
i19 = array[:,19]
i20 = array[:,20]
i21 = array[:,21]
i22 = array[:,22]
i23 = array[:,23]
i24 = array[:,24]
i25 = array[:,25]
i26 = array[:,26]
i27 = array[:,27]
i28 = array[:,28]

i_1=pd.DataFrame(i1)
i_2=pd.DataFrame(i2)
i_3=pd.DataFrame(i3)
i_4=pd.DataFrame(i4)
i_5=pd.DataFrame(i5)
i_6=pd.DataFrame(i6)
i_7=pd.DataFrame(i7)
i_8=pd.DataFrame(i8)
i_9=pd.DataFrame(i9)
i_10=pd.DataFrame(i10)
i_11=pd.DataFrame(i11)
i_12=pd.DataFrame(i12)
i_13=pd.DataFrame(i13)
i_14=pd.DataFrame(i14)
i_15=pd.DataFrame(i15)
i_16=pd.DataFrame(i16)
i_17=pd.DataFrame(i17)
i_18=pd.DataFrame(i18)
i_19=pd.DataFrame(i19)
i_20=pd.DataFrame(i20)
i_21=pd.DataFrame(i21)
i_22=pd.DataFrame(i22)
i_23=pd.DataFrame(i23)
i_24=pd.DataFrame(i24)
i_25=pd.DataFrame(i25)
i_26=pd.DataFrame(i26)
i_27=pd.DataFrame(i27)
i_28=pd.DataFrame(i28)

s_input = pd.concat([i_1, i_2, i_3, i_4, i_5, i_6, i_7, i_8, i_9, i_10, i_11, i_12, i_13, i_14, i_15, i_16, i_17, i_18, i_19, i_20, i_21, i_22, i_23, i_24, i_25, i_26, i_27, i_28], axis=1)
#print(s_input)
s_output = s_data.loc[:,'bug']
#print(s_output)

# SMOTE balancing method
from collections import Counter
print('Original source dataset shape %s' % Counter(s_output))
from imblearn.over_sampling import SMOTE
sm = SMOTE()
s_input, s_output = sm.fit_resample(s_input, s_output)
print('Resampled source dataset shape %s' % Counter(s_output))

# Normalization
#from sklearn import preprocessing
#s_normalized_input = preprocessing.normalize(s_input,norm='l2')
s_normalized_input = s_input

# Load target data
t_data = pd.read_csv("F:\\HDP\\data\\NASA\\pc4.csv")
#t_data = pd.read_csv("F:\\oodata_python\\jedit\\jedit-4.2.csv")

array1 = t_data.values
j1 = array1[:,27]
j2 = array1[:,30]
j3 = array1[:,5]
j4 = array1[:,9]
j5 = array1[:,4]
j6 = array1[:,31]
j7 = array1[:,32]
j8 = array1[:,35]
j9 = array1[:,23]
j10 = array1[:,34]
j11 = array1[:,17]
j12 = array1[:,24]
j13 = array1[:,25]
j14 = array1[:,15]
j15 = array1[:,19]
j16 = array1[:,33]
j17 = array1[:,21]
j18 = array1[:,36]
j19 = array1[:,2]
j20 = array1[:,28]
j21 = array1[:,12]
j22 = array1[:,0]
j23 = array1[:,8]
j24 = array1[:,20]
j25 = array1[:,3]
j26 = array1[:,7]
j27 = array1[:,26]
j28 = array1[:,29]

j_1=pd.DataFrame(j1)
j_2=pd.DataFrame(j2)
j_3=pd.DataFrame(j3)
j_4=pd.DataFrame(j4)
j_5=pd.DataFrame(j5)
j_6=pd.DataFrame(j6)
j_7=pd.DataFrame(j7)
j_8=pd.DataFrame(j8)
j_9=pd.DataFrame(j9)
j_10=pd.DataFrame(j10)
j_11=pd.DataFrame(j11)
j_12=pd.DataFrame(j12)
j_13=pd.DataFrame(j13)
j_14=pd.DataFrame(j14)
j_15=pd.DataFrame(j15)
j_16=pd.DataFrame(j16)
j_17=pd.DataFrame(j17)
j_18=pd.DataFrame(j18)
j_19=pd.DataFrame(j19)
j_20=pd.DataFrame(j20)
j_21=pd.DataFrame(j21)
j_22=pd.DataFrame(j22)
j_23=pd.DataFrame(j23)
j_24=pd.DataFrame(j24)
j_25=pd.DataFrame(j25)
j_26=pd.DataFrame(j26)
j_27=pd.DataFrame(j27)
j_28=pd.DataFrame(j28)

t_input = pd.concat([j_1, j_2, j_3, j_4, j_5, j_6, j_7, j_8, j_9, j_10, j_11, j_12, j_13, j_14, j_15, j_16, j_17, j_18, j_19, j_20, j_21, j_22, j_23, j_24, j_25, j_26, j_27, j_28], axis=1)
#print(s_input)
t_output = t_data.loc[:,'bug']
#print(s_output)

# SMOTE balancing method
from collections import Counter
print('Original target dataset shape %s' % Counter(t_output))
from imblearn.over_sampling import SMOTE
sm = SMOTE()
t_input, t_output = sm.fit_resample(t_input, t_output)
print('Resampled target dataset shape %s' % Counter(t_output))

# Normalization
#from sklearn import preprocessing
#t_normalized_input = preprocessing.normalize(t_input,norm='l2')
t_normalized_input = t_input


# splitting the testing data sets
from sklearn.model_selection import train_test_split
t_input_train, t_input_test, t_output_train, t_output_test = train_test_split(t_normalized_input, t_output, test_size=0.3)


from sklearn.tree import DecisionTreeClassifier 
#from sklearn import metrics

# Building decision tree model
# Create Decision Tree classifer object
c = DecisionTreeClassifier()
# Train Decision Tree Classifer
c = c.fit(s_input,s_output)
#Predict the response for test dataset
t11_output_train = c.predict(t_input_train)



# Length of target data
len_tdata = len(t_data)
print('Length of target data: \n',len_tdata)

new_source_input_data = np.concatenate((s_normalized_input, t_input_train), axis=0)

new_source_output_data = np.concatenate((s_output, t_output_train), axis=0)

new_input_test = t_input_test
new_output_test = t_output_test

# splitting the new_source_input and output data
from sklearn.model_selection import train_test_split
t1_input_train, t1_input_test, t1_output_train, t1_output_test = train_test_split(new_source_input_data, new_source_output_data, test_size=0.2)

new_input_val = t1_input_test
new_output_val = t1_output_test

import numpy as np 
mu, sigma = 0, 0.1 

l = len(t1_input_train)
# creating a noise with the same dimension as the dataset (2,2) 
noise = np.random.normal(mu, sigma, [l,28]) 
#print(noise)

new_input_train = (t1_input_train+noise)
new_output_train = t1_output_train




#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
#Create a Gaussian Classifier
gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(new_input_train,new_output_train)
#Predict the response for test dataset
y_pred = gnb.predict(new_input_test)
#print(y_pred)

# Confusion matrix generation
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(new_output_test, y_pred)
print('Confusion Matrix: \n',cf)

# Accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(new_output_test, y_pred)
print('Accuracy: %f'%acc)

# Recall
from sklearn.metrics import recall_score
re = recall_score(new_output_test, y_pred)
print('Recall: %f' % re)

# Precision
from sklearn.metrics import precision_score
pr = precision_score(new_output_test, y_pred)
print('Precision: %f' % pr)

# F1 Score
from sklearn.metrics import f1_score
f1 = f1_score(new_output_test, y_pred)
print('F1 score: %f' % f1)

# ROC Curve and AUC
from sklearn.metrics import roc_auc_score
# ROC AUC
auc = roc_auc_score(new_output_test, y_pred)
print('ROC AUC: %f' % auc)

tn, fp, fn, tp = confusion_matrix(new_output_test, y_pred).ravel()

far = (fp)/(fp+tn)
print('False Alaram/Positive Rate: %f' % far)

