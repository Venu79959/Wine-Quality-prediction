#!/usr/bin/env python
# coding: utf-8

# # Wine quality prediction using ML by KODI VENU

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv('winequality-red.csv')


# Top 5 rows

# In[2]:


data.head()


# Last 5 rows

# In[3]:


data.tail()


# Dataset Shape

# In[4]:


data.shape


# In[5]:


print('Number of rows', data.shape[0])
print('Number of columns', data.shape[1])


# Dataset Information

# In[6]:


data.info()


# Check null values in the dataset

# In[7]:


data.isnull().sum()


# dataset statistics

# In[8]:


data.describe()


# Quality vs fixed acidity

# In[9]:


data.columns


# In[10]:


plt.bar(data['quality'],data['fixed acidity'])
plt.xlabel('Quality')
plt.ylabel('fixed acidity')
plt.show()


# Volatile acidity vs quality

# In[11]:


plt.bar(data['quality'],data['volatile acidity'])
plt.xlabel('Quality')
plt.ylabel('volatile acidity')
plt.show()


# Residual sugar vs quality

# In[12]:


data.columns


# In[13]:


plt.bar(data['quality'],data['residual sugar'])
plt.xlabel('quality')
plt.ylabel('residual sugar')
plt.show()


# chlorides vs quality

# In[14]:


plt.bar(data['quality'],data['chlorides'])
plt.xlabel('quality')
plt.ylabel('chlorides')
plt.show()


# Total sulfur dioxide vs quality

# In[16]:


plt.bar(data['quality'],data['total sulfur dioxide'])
plt.xlabel('quality')
plt.ylabel('total sulfur dioxide')
plt.show()


# alcohol vs quality

# In[17]:


plt.bar(data['quality'],data['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# correlation matrix

# In[18]:


import seaborn as sns
sns.heatmap(data.corr(),annot=True)
plt.figure(figsize=(10,5))
sns.heatmap(data.corr(),annot=True,fmt='0.1f')


# Binorization of target variable

# In[19]:


data['quality'].unique()


# In[20]:


data['quality']=[1 if x>=7 else 0 for x in data['quality']]
data['quality'].unique()


# Not handling imbalanced dataset

# In[21]:


data['quality'].value_counts()


# In[22]:


sns.countplot(data['quality'])


# store feature matrix in X & response (target) in vector y

# In[23]:


X=data.drop('quality',axis=1)
y=data['quality']


# Train/test split

# In[24]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# Feature scaling

# In[25]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
X_train


# Applying PCA

# In[26]:


from sklearn.decomposition import PCA
pca=PCA(n_components=0.90)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)


# Logistic regression

# In[27]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
y_pred1=log.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)


# In[28]:


pip install -U imbalanced-learn


# Handling imbalanced dataset

# In[29]:


from imblearn.over_sampling import SMOTE


# In[30]:


X_res,y_res=SMOTE().fit_resample(X,y)
y_res.value_counts()


# Again train/test split

# In[31]:


X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.20,random_state=42)


# Again applying PCA

# In[32]:


from sklearn.decomposition import PCA
pca=PCA(n_components=0.90)
X_train=pca.fit_transform(X_train)
X_test=pca.transform(X_test)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)


# Apply logistic regression again

# In[33]:


from sklearn.linear_model import LogisticRegression
log=LogisticRegression()
log.fit(X_train,y_train)
y_pred1=log.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred1)
from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y_test,y_pred1)
recall_score(y_test,y_pred1)
f1_score(y_test,y_pred1)


# SVC

# In[34]:


from sklearn import svm
svm=svm.SVC()
svm.fit(X_train,y_train)
y_pred2=svm.predict(X_test)
accuracy_score(y_test,y_pred2)
precision_score(y_test,y_pred2)
recall_score(y_test,y_pred2)
f1_score(y_test,y_pred2)


# Kneighbors classifier

# In[35]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred3=svm.predict(X_test)
accuracy_score(y_test,y_pred3)
precision_score(y_test,y_pred3)
recall_score(y_test,y_pred3)
f1_score(y_test,y_pred3)


# Decision Tree classifier

# In[38]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
y_pred4=dt.predict(X_test)
accuracy_score(y_test,y_pred4)
precision_score(y_test,y_pred4)
recall_score(y_test,y_pred4)
f1_score(y_test,y_pred4)


# Random forest classifier

# In[39]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred5=svm.predict(X_test)
accuracy_score(y_test,y_pred5)
precision_score(y_test,y_pred5)
recall_score(y_test,y_pred5)
f1_score(y_test,y_pred5)


# Gradient boosting classifier

# In[40]:


from sklearn.ensemble import GradientBoostingClassifier
gr=GradientBoostingClassifier()
gr.fit(X_train,y_train)
y_pred6=svm.predict(X_test)
accuracy_score(y_test,y_pred6)
precision_score(y_test,y_pred6)
recall_score(y_test,y_pred6)
f1_score(y_test,y_pred6)


# In[41]:


import pandas as pd


# In[42]:


final_data=pd.DataFrame({'Models':['LR','SVC','KNN','DT','RF','GR'],'ACC':[accuracy_score(y_test,y_pred1)*100,accuracy_score(y_test,y_pred2)*100,accuracy_score(y_test,y_pred3)*100,accuracy_score(y_test,y_pred4)*100,accuracy_score(y_test,y_pred5)*100,accuracy_score(y_test,y_pred6)*100]})


# In[43]:


final_data


# In[46]:


import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
sns.barplot(final_data['Models'],final_data['ACC'])


# Save the model

# In[47]:


X=data.drop('quality',axis=1)
y=data['quality']


# In[50]:


from imblearn.over_sampling import SMOTE
X_res,y_res=SMOTE().fit_resample(X,y)
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
X=st.fit_transform(X_res)
X=pca.fit_transform(X)


# In[51]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X,y_res)


# In[53]:


import joblib
joblib.dump(rf,"wine_quality_prediction")
model=joblib.load('wine_quality_prediction')


# Prediction on New Data

# In[57]:


import pandas as pd
new_data=pd.DataFrame({'fixed acidity':7.3,'volatile acidity':0.65,'citric acid':0.00,'residual sugar':1.2,'chlorides':0.065,'free sulfur dioxide':15.0,'total sulfur dioxide':21.0,'density':0.9946,'pH':3.39,'sulphates':0.47,'alcohol':10.0},index=[0])


# In[58]:


new_data


# In[59]:


test=pca.transform(st.transform(new_data))


# In[60]:


p=model.predict(test)


# In[61]:


p


# In[62]:


if p[0]==1:
    print('Good Quality Wine')
else:
    print("Bad Quality Wine")


# In[ ]:




