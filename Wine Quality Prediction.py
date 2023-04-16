# Import required libraris
import numpy as np
import pandas as pd

# Loading wine quality dataset
df=pd.read_csv('Wine-Quality-Dataset.csv')

# Create Classification version of target variable
df['goodquality'] = [1 if x >= 7 else 0 for x in df['quality']]

# Seperate Feature variable and target variable
X=df.drop(['quality','goodquality'],axis=1)
y=df['goodquality']

# Standardizing the feature variables
from sklearn.preprocessing import StandardScaler

X_feature = X
X = StandardScaler().fit_transform(X)

# ## Spliting and applying algorithms

# In[17]:


# Spliting the data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

# In[18]:


from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(random_state=1)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)

print(classification_report(y_test, tree_pred))

# In[19]:


from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(random_state=1)
RF_model.fit(X_train, y_train)
RF_pred = RF_model.predict(X_test)

print(classification_report(y_test, RF_pred))

# In[20]:


from sklearn.ensemble import AdaBoostClassifier

adbc_model = AdaBoostClassifier(random_state=1)
adbc_model.fit(X_train, y_train)
adbc_pred = adbc_model.predict(X_test)

print(classification_report(y_test, adbc_pred))

# In[21]:


from sklearn.ensemble import GradientBoostingClassifier

gbc_model = GradientBoostingClassifier(random_state=1)
gbc_model.fit(X_train, y_train)
gbc_pred = gbc_model.predict(X_test)

print(classification_report(y_test, gbc_pred))

# In[22]:


from xgboost import XGBClassifier

XG_model = XGBClassifier(random_state=1)
XG_model.fit(X_train, y_train)
XG_pred = XG_model.predict(X_test)

print(classification_report(y_test, XG_pred))

# ## Finding Important Featues

# In[23]:


feature_importance = pd.Series(RF_model.feature_importances_, index=X_feature.columns)
feature_importance.nlargest(25).plot(kind='barh', figsize=(10, 5))

# In[24]:


feature_importance = pd.Series(XG_model.feature_importances_, index=X_feature.columns)
feature_importance.nlargest(25).plot(kind='barh', figsize=(10, 5))

# In[25]:


import pickle

# Serialize and save the model to a file
with open('RF_model.pkl', 'wb') as file:
    pickle.dump(XG_model, file)

# #Load the serialized model from file
# with open('my_model.pkl', 'rb') as file:
#     loaded_model = pickle.load(file)

