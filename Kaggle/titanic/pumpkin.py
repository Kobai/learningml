import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("train.csv")

survived_class = df[df['Survived']==1]['Pclass'].value_counts()
dead_class = df[df['Survived']==0]['Pclass'].value_counts()
df_class = pd.DataFrame([survived_class,dead_class])
df_class.index=['Survived','Died']
df_class.plot(kind='bar', stacked=True, figsize=(5,3), title="Survived/Died by Class")

Class1_survived = df_class.iloc[0,0]/df_class.iloc[:,0].sum()*100
Class2_survived = df_class.iloc[0,1]/df_class.iloc[:,1].sum()*100
Class3_survived = df_class.iloc[0,2]/df_class.iloc[:,2].sum()*100
print(round(Class1_survived))
print(round(Class2_survived))
print(round(Class3_survived))
print(df_class)


Survived = df[df['Survived']==1]['Sex'].value_counts()
Died = df[df['Survived']==0]['Sex'].value_counts()
df_sex= pd.DataFrame([Survived, Died])
df_sex_index = ['Survived','Died']
df_sex.plot(kind='bar', stacked = True, figsize=(5,3), title="Survived/ Died by Sex")
print(df_sex)
female_survived = df_sex.female.iloc[0]/df_sex.female.sum() * 100
male_survived = df_sex.male.iloc[0]/df_sex.male.sum() * 100
print(round(female_survived))
print(round(male_survived))


plt.show()


