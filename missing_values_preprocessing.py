#!/usr/bin/env python
# coding: utf-8

# ## Traitement des valeurs manquantes
# 

# Dans une analyse predictive de données, l'une des tâches que nous deverios effectuer avant la formatin de notre modèle d'apprentissage automatique est le preprocessing des données. Le nettoyage des données est un élément clé de la tâche de preprocessing des données et implique généralement la suppression des valeurs manquantes ou leur remplacement par la moyenne, la mediane, la mode ou une constante.

# ### **Pourquoi faut-il remplir les données manquantes ?** 
# 
# 1. La plupart des modèles d'apprentissage automatique généreront une erreur si on leur transmet des valeurs NaN. 
# 2. Le moyen le plus simple consiste simplement à les remplir avec 0, mais cela peut réduire considérablement la précision du modèle.
# 3. Pour remplir les valeurs manquantes, il existe de nombreuses méthodes disponibles. 
# 4. Pour choisir la meilleure méthode, on doit comprendre tout d'abord le type de valeurs manquantes et leur signification.

# On trouve généralement les valeurs manquantes sous forme de **NaN** ou **null** ou **None** dans le jeu de données.
# 
# **Nous allons télecharcher notre jeu de données que nous avons récuperé sur Kaggle via lien ci-dessous :**
# 
# https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset?select=test_Y3wMUE5_7gLdaTN.csv

# In[1]:


import pandas as pd
import numpy as np
 
train_data = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
#test_data = pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
#data = pd.concat([train_data, test_data], ignore_index=True)


# In[2]:


# Affichons l'entête de données
train_data.head()


# In[3]:


#Afficher la forme de données
train_data.shape


# In[4]:


train_data["Loan_Status"].value_counts()


# Nous avons 981 lignes et 13 colonnes dans notre jeu de données

# In[5]:


# Pour avoir des informations sur notre jeu des données
train_data.info()


# On constate qu'il y'a 7 colonnes qui contiennent des valeurs manquantes dont 5 des colonnes sont des caractéristiques catégorielles et 3 des caractéristiques numériques. 
# 

# In[6]:


# Pour avoir le total de valeurs manquantes par colonne
train_data.isnull().sum()


# On constate que la caractéristique **Credit_History** est celle qui a le plus de valeurs manquantes (50)

# Vérifieons qu'il existe également des valeurs catégorielles dans l'ensemble de données. Pour cela, nous devons utiliser **Label Encoding** ou **One Hot Encoding**.

# In[7]:


train_data.columns


# #### **Les méthodes de gestion de valeurs manquantes**
# 
# 1. Suppression des colonnes avec des données manquantes
# 2. Suppression des lignes avec des données manquantes
# 3. Remplir les données manquantes avec une valeur : imputation
# 4. Remplir avec un modèle de régression

# ### **1. Suppression des colonnes avec des données manquantes**
# 
# Dans ce cas, supprimons les colonnes avec des valeurs manqantes, puis ajustons le modèle et vérifions sa précision.
# 
# Mais il s'agit d'un cas extrême et ne doit être utilisé que lorsqu'il existe de nombreuses valeurs manqantes dans la colonne.

# In[8]:


train_data_without_missing_axis1=train_data.dropna(axis=1)
train_data_without_missing_axis1.isnull().sum()


# https://stackoverflow.com/questions/24458645/label-encoding-across-multiple-columns-in-scikit-learn

# #### **One Hot Encoding**
# 
# Vérifions qu'il existe également des valeurs catégorielles dans l'ensemble de données. Pour cela, vous devons utiliser **Label Encoding** ou **One Hot Encoding**.

# In[11]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_data_without_missing_axis1=train_data_without_missing_axis1.apply(LabelEncoder().fit_transform)
print(train_data_without_missing_axis1.head())
print()
print(train_data_without_missing_axis1)


# In[12]:


X_without_missing_axis1 =train_data_without_missing_axis1.drop(["Loan_Status","Loan_ID"], axis=1)

y=train_data_without_missing_axis1["Loan_Status"]


# In[13]:


y.value_counts()


# In[14]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X_without_missing_axis1,y,test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print(round(metrics.accuracy_score(pred,y_test), 4))


# Cette méthode de gestion des valeurs manquantes donne un classificateur moins précis. Nous avons une précision de **64.86%**. Nous allons utiliser d'autres méthodes pour y voir claire

# ### **2. Supprimer les lignes avec des données manquantes**
# 
# S'il y a une certaine ligne avec des données manquantes, nous pouvez supprimer la ligne entière avec toutes les entités de cette ligne.
# 
# **axis=1 :** est utilisé pour supprimer la colonne avec les valeurs `NaN`.
# 
# **axis=0 :** est utilisé pour supprimer la ligne avec les valeurs `NaN`.

# In[15]:


train_data_without_missing_axis0=train_data.dropna(axis=0)
train_data_without_missing_axis0.isnull().sum()


# In[16]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train_data_without_missing_axis0=train_data_without_missing_axis0.apply(LabelEncoder().fit_transform)
print(train_data_without_missing_axis0.head())
print()
print(train_data_without_missing_axis0)


# In[17]:


X_without_missing_axis0 =train_data_without_missing_axis0.drop(["Loan_Status","Loan_ID"], axis=1)

y=train_data_without_missing_axis0["Loan_Status"]

y.value_counts()


# In[18]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X_without_missing_axis0,y,test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print(round(metrics.accuracy_score(pred,y_test), 4))


# Supprimer les lignes contenant des valeurs manquantes, c'est une meilleure façon de gérer les valeurs manquantes que de supprimer les colonnes avec des valeurs manquantes, car la précision de notre modéle est passée de 64.86% à 79.86%. 
# 
# **NB :** Les colonnes suprimées ont beacoup plus d'informations que prévu.

# ### **3. Remplir les valeurs manquantes : imputation**
# 
# Dans ce cas, nous remplirons les valeurs manquantes avec un certain nombre.
# 
# Les manières possibles de le faire sont :
# 
# 1. Remplir les données manquantes avec la valeur moyenne ou médiane s'il s'agit d'une variable numérique.
# 2. Remplir les données manquantes avec la mode s'il s'agit d'une variable catégorielle.
# 3. Remplir la valeur numérique avec 0 ou -999, ou un autre nombre qui n'apparaîtra pas dans les données. Cela peut être fait pour que la machine puisse reconnaître que les données ne sont pas réelles ou sont différentes.
# 4. Remplir la variable catégorielle avec un nouveau type pour les valeurs manquantes.
# 
# Nous pouvons utiliser la fonction pandas **fillna()** pour remplir les valeurs manquantes dans l'ensemble de données.

# ### **Remplir les données manquantes avec la valeur moyenne ou médiane s'il s'agit d'une variable numérique et la mode s'il s'agit d'une variable catégorielle**
# 
# 

# In[19]:


# Fonction pour remplir les données manquantes
def transform_features(df):
    
    df =df.fillna(df.select_dtypes(include=['object']).mode().iloc[0], axis=0)
    
    df = df.fillna(df.select_dtypes(include=['int', 'float']).mean(), axis=0)

    return df


# In[20]:


transform_df=transform_features(train_data)

print(transform_df.isnull().sum())


# In[21]:


transform_df.shape


# In[22]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

transform_df=transform_df.apply(LabelEncoder().fit_transform)
print(transform_df.head())
print()
print(transform_df.shape)


# In[23]:


X=transform_df.drop(["Loan_Status","Loan_ID"], axis=1)

y=transform_df["Loan_Status"]

y.value_counts()


# In[24]:


from sklearn import metrics
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print(round(metrics.accuracy_score(pred,y_test), 4))


# La valeur de précision ressort à 78,92%, ce qui est une réduction par rapport au cas précédent.
# 
# Cela ne se produira pas en général, dans ce cas, cela signifie que la moyenne et la mode n'ont pas pu remplir correctement les valeurs manquantes.  

# ### **Références pour aller plus loin**
# 
# 
# https://towardsdatascience.com/imputing-missing-values-using-the-simpleimputer-class-in-sklearn-99706afaff46
# 
# https://vitalflux.com/pandas-impute-missing-values-mean-median-mode/
# 
# https://www.analyticsvidhya.com/blog/2021/05/dealing-with-missing-values-in-python-a-complete-guide/
# 
# https://towardsdatascience.com/pandas-tricks-for-imputing-missing-data-63da3d14c0d6
# 
# 
# https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html

# In[ ]:




