# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

File_Path = 'C:/Users/User/Downloads/'
File_Name = 'car_data.csv'

df = pd.read_csv(File_Path + File_Name)

df.drop(columns=['User ID'],inplace=True)
encoders = []
for i in range(0,len(df.columns)-1):
              enc = LabelEncoder()
              df.iloc[:,i] = enc.fit_transform(df.iloc[:,i])
              encoders.append(enc)

x = df.iloc[:,0:4]
y = df['Purchased']
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(x,y)

x_pred = ['Male','40',"107000"]

for i in range(0,len(df.columns)-1):
    x_pred[i] = encoders[i].transform([x_pred[i]])
    
x_pred_adj = np.array(x_pred).reshape(-1,4)

y_pred = model.redict(x_pred_adj)
print('Prediction:',y_pred[0])
score = model.score(x,y)
print('Accuracy:','{:.2f}'.format(score))

feature = x.columns.tolist()
Data_class = y.tolist()


feature_importances = model.feature_importances_
feature_UserID = ['Gender','Age','AnnualSalary']
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x=feature_importances, y=feature_UserID)

print(feature_importances)

feature = x.columns.tolist()
Data_class  = y.tolist()

plt.figure(figsize=(25,20))
_=plot_tree(model,feature_UserID=feature,class_UserID=Data_class,label='all',impurity=True,precision=3,filled=True,rounded=True,fontsize=16)
plt.show()