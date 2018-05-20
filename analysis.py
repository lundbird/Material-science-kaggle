# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 19:42:12 2018

@author: Alex Lundberg and Yaoyue Zhou

Kaggle competition script for analyzing material science data. 
Predicts bandgap energy and formation energy.
"""
#setup imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#data pre prep
df = pd.read_csv("data.csv")
del df['entry']
df.columns=['BandgapE','dataset','FormationE','id','alpha','beta','gamma','a','b','c','atomcount','al','ga','in','spacegroup']
df['spacegroup'] = df['spacegroup'].astype('category')
train = df[df["dataset"]=="train"].copy()
test = df[df['dataset']=="test"].copy() #wont need test dataset because competition is closed
del train['dataset']
del train['id']

#angles for most spacegroups are consistent
#vectors are moderately consistent
#energies and element composition are variable.
train.groupby('spacegroup').std()/train.groupby('spacegroup').mean()

#lets take a close look at the 12 and 227 spacegroups. May be composed of 
#subgroups based on angle variability
train[train['spacegroup']==12].hist(column=['alpha','beta','gamma']) #12 is of one group
train[train['spacegroup']==227].hist(column=['alpha','beta','gamma']) #2 subgroups based on gamma!
train[(train['spacegroup']==227) & (train['gamma']<80)].gamma.hist()
train[(train['spacegroup']==227) & (train['gamma']>80)].gamma.hist() #confirmed
train[(train['spacegroup']==227) & (train['gamma']<80)].std()/train[(train['spacegroup']==227) & (train['gamma']<80)].mean() #proof

#now that we know that 227 has two subgroups lets adjust the spacegroup field for our dataframe
train['spacegroup'] = train['spacegroup'].astype('float64')
train.loc[(train['spacegroup']==227) & (train['gamma']>80),'spacegroup'] = 228 #make new category 228
train['spacegroup'] = train['spacegroup'].astype('category') #reset to category


#plot heatmap of correlations
sns.heatmap(train.corr()) 
#Low correlation for most variables except for aluminum + and indium - correlation to BandgapE

#The density and/or volume of the structure may be useful. Lets find it.
def get_vol(a, b, c, alpha, beta, gamma):
    return a*b*c*np.sqrt(1 + 2*np.cos(alpha)*np.cos(beta)*np.cos(gamma)
                           - np.cos(alpha)**2
                           - np.cos(beta)**2
                           - np.cos(gamma)**2)

lattice_angles = ['alpha', 'beta', 'gamma']
for lang in lattice_angles:
    train[lang] = np.pi * train[lang] / 180
    
# compute the cell volumes 
train['vol'] = get_vol(train['a'], train['b'], train['c'],
                          train['alpha'], train['beta'], train['gamma'])

#compute the atomic density
train['atomic_density'] = train['atomcount'] / train['vol']   

#compute weight density
def get_weight_density(al, ga, ind, vol):
    return (al*27.0 + ga*69.7 + ind*114.8) / vol

train['weight_density'] = get_weight_density(train['al'],train['ga'],train['in'],train['vol'])

#lets try the headmap again.
sns.heatmap(train.corr())

#seems like atomic density is a good predictor.
#Lets try engineering more predictors

#now we want to get the xyz data which gives the locations of the atoms and oxygens in the unit cell
'''
def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float),x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)

#for each row in the dataframe we have the molecular configurations file.
#we need to extract useful features from it and then put it back into the dataframe.

for x in range(1,2400):
    fn = "train/train/{}/geometry.xyz".format(x)
    crystal_xyz, crystal_lat = get_xyz_data(fn)
    
    #we can just throw out the crystal_lat field as it contains the same information as the
    #lattice parameters and the angle since that is already included in the original csv.
    
    #lets engineer a feature Oxygen count, oxygen density, metal count, metal density
    A = np.transpose(crystal_lat)
    vol = np.linalg.det(A)
    #extracts count of each atom in the set
    natom = len(crystal_xyz)
    m_atoms = [i for i in range(natom) if crystal_xyz[i][1] != 'O']
    o_atoms = [i for i in range(natom) if crystal_xyz[i][1] == 'O']
    
    train.at[x-1,'o_atoms'] = len(o_atoms)
    train.at[x-1,'m_atoms'] = len(m_atoms)
    
#Lets check our correlations again
train = train.fillna(train.mean())  #fill null values
sns.heatmap(train.corr())
#This factor doesnt seem to influence our desired variable.
'''
#this is enough feature engineering for now. We can get alot more in depth with graph theory and linear algebra
#for more info see source https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates

#dummify the categorical variables
categories = train.dtypes[train.dtypes=='category'] 
categoricals = pd.get_dummies(train[categories.index])

#features to add --Formation energy, band gap, average ionization potential
#lets start our Machine Learning

#spit into tain and test sets
features = ['BandgapE','FormationE']
Bandgap = train['BandgapE'].values
Formation = train['FormationE'].values

predictors = [col for col in list(train) if col not in features]
X = train[predictors]
Bandgap_train, Bandgap_test, Bandgap_train, Bandgap_test = train_test_split(X, Bandgap, test_size=0.3, random_state=42)
Formation_train, Formation_test, Formation_train, Formation_test = train_test_split(X, Formation, test_size=0.3, random_state=42)


#Model choosing: Can pick another if desired or adjust hyperparameters
model = RandomForestRegressor()

model_Formation = model #fit the data to a model and find error
model_Bandgap = model #fit the data to a model and find error
model_Formation.fit(X,Formation)
model_Bandgap.fit(X,Bandgap)
model_Formation.fit(X,Formation)

importances_E =  model_Bandgap.feature_importances_
importances_Eg =  model_Bandgap.feature_importances_

#shows most important features for bandgap and formation
descending_indices_E = np.argsort(importances_E)[::-1]
sorted_importances_E = [importances_E[idx] for idx in descending_indices_E]
sorted_features_E = [predictors[idx] for idx in descending_indices_E]
print('most important feature for formation energy is %s' % sorted_features_E[0])

# collect ranking of most "important" features for Eg
descending_indices_Eg = np.argsort(importances_Eg)[::-1]
sorted_importances_Eg = [importances_Eg[idx] for idx in descending_indices_Eg]
sorted_features_Eg = [predictors[idx] for idx in descending_indices_Eg]
print('most important feature for band gap is %s' % sorted_features_Eg[0])

#Output cross val scores
scores = cross_val_score(model, X, Bandgap, cv=5)

print("here are the final scores for Bandgap")
print(scores)

print("average cross val score")
print(np.mean(scores))

scores = cross_val_score(model, X, Formation, cv=5)
print("here are the final scores for Formation Energy")
print(scores)

print("average cross val score")
print(np.mean(scores))


#TODO three different models, cross val score, hyperparameter adjustment.


#TODO PCA on the most important features.







        

















