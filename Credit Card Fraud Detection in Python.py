#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud Detection Project

# ### Import the necessary modules

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools   # advanced tools
import warnings 
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler   # data normalization
from sklearn.model_selection import train_test_split   # data split
from sklearn.ensemble import RandomForestClassifier   # Random forest tree algorithm
from sklearn.tree import DecisionTreeClassifier    # Decision tree algorithm
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score   # evaluation metric


# ### Load the csv file

# In[17]:


df = pd.read_csv("creditcard.csv")
df.head()


# ### Perform Exploratory Data Analysis

# In[3]:


df.info()


# In[4]:


df.isnull().values.any()


# In[5]:


df["Amount"].describe()


# In[6]:


cases = len(df)
non_fraud = len(df[df.Class == 0])
fraud = len(df[df.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100

from termcolor import colored as cl

print(cl('CASE COUNT', attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('Total number of cases are {}'.format(cases), attrs = ['bold']))
print(cl('Number of Non-fraud cases are {}'.format(non_fraud), attrs = ['bold']))
print(cl('Number of Non-fraud cases are {}'.format(fraud), attrs = ['bold']))
print(cl('Percentage of fraud cases is {:.4f}'.format(fraud_percent), attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))

# 0 means non fraud; 1 means fraud


# In[7]:


# Visualize the "Labels" column in our dataset

labels = ["Genuine", "Fraud"]
count_classes = df.value_counts(df['Class'], sort = True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# In[8]:


nonfraud_cases = df[df.Class == 0]
fraud_cases = df[df.Class == 1]

print(cl('CASE AMOUNT STATISTICS', attrs = ['bold']))
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('NON-FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(nonfraud_cases.Amount.describe())
print(cl('--------------------------------------------', attrs = ['bold']))
print(cl('FRAUD CASE AMOUNT STATS', attrs = ['bold']))
print(fraud_cases.Amount.describe())
print(cl('--------------------------------------------', attrs = ['bold']))


# In[9]:


import seaborn as sns

df["Group"] = np.where(df["Class"] == 0, "Valid", "Fraud")

fig, ax = plt.subplots(1, 1, figsize = (10,5))
sns.boxplot(x = "Group", y = "Amount", linewidth = 0.5, width = 0.8, data = df)
plt.title("Distribution of transaction amount by class", y = 1.05, x = 0.5)
plt.show()


# ### Correlation matrix

# In[10]:


fig = plt.figure(figsize = (15, 15))
sns.heatmap(df.corr(), vmax = 1, linewidths = 0.01, square = True, annot = True, cmap = 'viridis', linecolor = "white")
plt.show()


# In[11]:


fig,axes = plt.subplots(2,1, figsize = (15,10))
sns.distplot(df.query("Group == 'Valid'").Time, color = "blue", ax = axes[0], kde = False, bins = 100, label = "Valid") 
sns.distplot(df.query("Group == 'Fraud'").Time, color = "red", ax = axes[1], kde = False, bins = 100, label = "Fraud")
fig.legend(loc = 'center right')
plt.title("Distribution of time of transaction by class", y = 2.3)
plt.show()


# ### Perform Scaling

# In[18]:


scaler = StandardScaler()
df["NormalizedAmount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df.drop(["Amount", "Time"], inplace = True, axis = 1)

Y = df["Class"]
X = df.drop(["Class"], axis = 1)


# In[19]:


Y.head()


# ### Split the data

# In[20]:


(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size = 0.3, random_state = 42)

print("Shape of train_X: ", train_X.shape)
print("Shape of test_X: ", test_X.shape)


# Let's train different models on our dataset and observe which algorithm works better for our problem.
# 

# ### Decision Tree Classifier

# In[21]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)

predictions_dt = decision_tree.predict(test_X)
decision_tree_score = decision_tree.score(test_X, test_Y) * 100


# ### Random Forest

# In[24]:


random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(train_X, train_Y)

predictions_rf = random_forest.predict(test_X)
random_forest_score = random_forest.score(test_X, test_Y) * 100


# ### K-Nearest Neighbors

# In[25]:


from sklearn.neighbors import KNeighborsClassifier # KNN algorithm

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_X, train_Y)

predictions_knn = knn.predict(test_X)
knn_score = knn.score(test_X, test_Y) * 100


# ### Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression # Logistic regression algorithm

lr = LogisticRegression()
lr.fit(train_X, train_Y)

predictions_lr = lr.predict(test_X)
lr_score = lr.score(test_X, test_Y) * 100


# ### SVM 

# In[27]:


from sklearn.svm import SVC # SVM algorithm

svm = SVC()
svm.fit(train_X, train_Y)

predictions_svm = svm.predict(test_X)
svm_score = svm.score(test_X, test_Y) * 100


# ### XGBoost

# In[28]:


from xgboost import XGBClassifier # XGBoost algorithm

xgb = XGBClassifier(n_estimators = 100)
xgb.fit(train_X, train_Y)

predictions_xgb = xgb.predict(test_X)
xgb_score = xgb.score(test_X, test_Y) * 100


# ### Print Scores of Our Classifiers

# In[29]:


print("Random Forest Score: ", random_forest_score)
print("Decision Tree Score: ", decision_tree_score)
print("KNN Score: ", knn_score)
print("Logistic Regression Score: ", lr_score)
print("SVM Score: ", svm_score)
print("XGBoost Score: ", xgb_score)


# ### Create a Function to Print the Metrics: Accuracy, Precision, Recall, and F1-Score

# In[30]:


from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score

def metrics(actuals, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))


# ### Visualize the Confusion Matrix and the Evaluation Metrics of Our Models

# In[31]:


confusion_matrix_dt = confusion_matrix(test_Y, predictions_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt)

sns.heatmap(confusion_matrix_dt, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - Decision Tree")

print("Evaluation of Decision Tree Model")
print()
metrics(test_Y, predictions_dt.round())


# In[32]:


confusion_matrix_rf = confusion_matrix(test_Y, predictions_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)

sns.heatmap(confusion_matrix_rf, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - Random Forest")

print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_rf.round())


# In[33]:


confusion_matrix_knn = confusion_matrix(test_Y, predictions_knn.round())
print("Confusion Matrix - KNN")
print(confusion_matrix_knn)

sns.heatmap(confusion_matrix_knn, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - KNN")

print("Evaluation of KNN Model")
print()
metrics(test_Y, predictions_knn.round())


# In[34]:


confusion_matrix_lr = confusion_matrix(test_Y, predictions_lr.round())
print("Confusion Matrix - Logistic Regression")
print(confusion_matrix_lr)

sns.heatmap(confusion_matrix_lr, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - Logistic Regression")

print("Evaluation of Logistic Regression Model")
print()
metrics(test_Y, predictions_lr.round())


# In[35]:


confusion_matrix_svm = confusion_matrix(test_Y, predictions_svm.round())
print("Confusion Matrix - SVM ")
print(confusion_matrix_svm)

sns.heatmap(confusion_matrix_svm, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - SVM")

print("Evaluation of SVM Model")
print()
metrics(test_Y, predictions_svm.round())


# In[36]:


confusion_matrix_xgb = confusion_matrix(test_Y, predictions_xgb.round())
print("Confusion Matrix - XGBoost")
print(confusion_matrix_xgb)

sns.heatmap(confusion_matrix_xgb, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - XGBoost")

print("Evaluation of XGBoost Model")
print()
metrics(test_Y, predictions_xgb.round())


# Clearly, XGBoost and Random Forest model work better than others.

# But, if we clearly observe our dataset suffers a serious problem of **class imbalance**. 
# The genuine (not fraud) transactions are more than 99% with the fraud transactions constituting of 0.17%.
# 
# With such kind of distribution, if we train our model without taking care of the imbalance issues, it predicts the label with higher importance given to genuine transactions (as there are more data about them) and hence obtains more accuracy.

# The class imbalance problem can be solved by various techniques. **Over sampling** is one of them.
#  
# One approach to addressing imbalanced datasets is to oversample the minority class. The simplest approach involves duplicating examples in the minority class, although these examples don’t add any new information to the model. 
# 
# Instead, new examples can be synthesized from the existing examples. This is a type of data augmentation for the minority class and is referred to as the **Synthetic Minority Oversampling Technique**, or **SMOTE** for short.

# ### Performing Oversampling on Random Forest and XGBoost

# In[37]:


from imblearn.over_sampling import SMOTE

X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of Y: ", Y_resampled.shape)

value_counts = Counter(Y_resampled)
print(value_counts)

(train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size= 0.3, random_state= 42)


# ### Build the Random Forest Classifier On the New Dataset

# In[38]:


rf_resampled = RandomForestClassifier(n_estimators = 100)
rf_resampled.fit(train_X, train_Y)

predictions_resampled = rf_resampled.predict(test_X)
random_forest_score_resampled = rf_resampled.score(test_X, test_Y) * 100


# ### Visualize the Confusion Matrix

# In[40]:


cm_resampled = confusion_matrix(test_Y, predictions_resampled.round())
print("Confusion Matrix - Random Forest")
print(cm_resampled)

sns.heatmap(cm_resampled, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - Random Forest After Oversampling")

print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_resampled.round())


# ### Build the XGBoost Classifier On the New Dataset

# In[41]:


xgb_resampled = XGBClassifier(n_estimators = 100)
xgb_resampled.fit(train_X, train_Y)

predictions_resampled_xgb = xgb_resampled.predict(test_X)
xgb_score_resampled = xgb_resampled.score(test_X, test_Y) * 100


# ### Visualize the Confusion Matrix

# In[42]:


cm_resampled_xgb = confusion_matrix(test_Y, predictions_resampled_xgb.round())
print("Confusion Matrix - XGBoost")
print(cm_resampled_xgb)

sns.heatmap(cm_resampled_xgb, annot = True, cmap = "YlOrRd", fmt = 'g')
plt.ylabel('Truth')
plt.xlabel('Predicted')
plt.title("Confusion Matrix - XGBoost After Oversampling")

print("Evaluation of XGBoost Model")
print()
metrics(test_Y, predictions_resampled_xgb.round())


# Now it is evident that after addressing the class imbalance problem, our Random Forest classifier with SMOTE performs far better than the Random Forest classifier withour SMOTE, but XGBoost classifier with SMOTE performs little worse than the XGBoost classifier withour SMOTE.

# In this project we have tried to show different methods of dealing with unbalanced datasets like the fraud credit card transaction dataset where the instances of fraudulent cases is few compared to the instances of normal transactions. 
# 
# 

# We concluded that the oversampling technique works best on the dataset and achieved significant improvement in model performance over the imabalanced data. The best score of 0.99967 was achieved using an XGBOOST model though Random Forest performed well too. It is likely that by further tuning the XGBOOST model parametres we can achieve even better performance. This project has demonstrated the importance of sampling effectively, modelling and predicting data with an imbalanced dataset.

# In[ ]:




