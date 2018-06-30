
# coding: utf-8

# In[1]:


import pandas as pd
import sklearn as sk


# In[3]:


df = pd.read_csv("C:/Users/Laptop/Python/BreastCancerData.csv", names=["ID", "Result","Mean_Radius","Mean_Texture","Mean_Perimeter",
                   "Mean_Area","Mean_Smoothness","Mean_Compactness", "Mean_Concavity",
                   "Mean_ConcavePoints", "Mean_Symmetry","Mean_FractalDimension",
                   "SE_Radius","SE_Texture","SE_Perimeter","SE_Area","SE_Smoothness",
                   "SE_Compactness","SE_Concavity","SE_ConcavePoints","SE_Symmetry",
                   "SE_FractalDimension","Worst_Radius","Worst_Texture","Worst_Perimeter",
                   "Worst_Area","Worst_Smoothness","Worst_Compactness","Worst_Concavity",
                   "Worst_ConcavePoints","Worst_Symmetry","Worst_FractalDimension"])


# In[4]:


df


# In[5]:


noID = df.drop('ID', axis=1)


# In[6]:


noID


# In[7]:


NoResults = noID.drop('Result', axis=1)


# In[8]:


NoResults


# In[31]:


from sklearn.preprocessing import StandardScaler


# In[38]:


FeatureScaling = lambda x: ((x - min(x)) / (max(x) - min(x))) 


# In[39]:


FeatureScaling


# In[40]:


BCD_Normalised = NoResults.apply(FeatureScaling)


# In[44]:


BCD_Normalised


# In[59]:


BCD_Training = BCD_Normalised[1:450]


# In[60]:


BCD_Training


# In[55]:


BCD_Test = BCD_Normalised[451:569]


# In[56]:


BCD_Test


# In[69]:


from sklearn.neighbors import KNeighborsClassifier


# In[73]:


import math


# In[76]:


K_Value = math.floor(math.sqrt(len(BCD_Training.index)))
knn = KNeighborsClassifier(n_neighbors=K_Value)


# In[77]:


from sklearn.model_selection import train_test_split


# In[83]:


X = BCD_Normalised.iloc[:, 0:].values
y = noID.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33)


# In[84]:


BCD_Prediction = knn.fit(X_train,y_train)


# In[85]:


predicting = knn.predict(X_test)


# In[86]:


from sklearn.metrics import classification_report, confusion_matrix


# In[87]:


print(confusion_matrix(y_test, predicting))


# In[89]:


print(classification_report(y_test, predicting))

