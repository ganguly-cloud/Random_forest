from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
np.random.seed(0)   # not mandetory

# ceating a n object called irir with the iris dada
iris = load_iris()

##print iris.data
'''
[6.5 3.  5.2 2. ]
 [6.2 3.4 5.4 2.3]'''

##print iris.feature_names
'''
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
'petal width (cm)']'''
##print iris.target
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]'''
##print iris.target_names
''' ['setosa' 'versicolor' 'virginica'] '''
# creating a dataframe with the four feature variables 
df = pd.DataFrame(iris.data,columns = iris.feature_names)

##print df.head()
print
'''
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2'''

# Adding a new column with the species name
df['species'] = pd.Categorical.from_codes(iris.target,iris.target_names)

#print len(df)   # 150

# Creating a test and Train Data

df['is_train'] = np. random.uniform(0,1,len(df)) <= .75
##print df['is_train']
##print df.head()
'''
   sepal length (cm)  sepal width (cm)  ...  species  is_train
0                5.1               3.5  ...   setosa      True
1                4.9               3.0  ...   setosa      True
2                4.7               3.2  ...   setosa      True
3                4.6               3.1  ...   setosa      True'''

# Creating  dataframes with test rows and train rows

train , test = df[df['is_train'] == True],df[df['is_train'] == False]

print 'Number of observations in the training data : ',len(train)   # 118
print 'Number of observations in the testing data : ',len(test)     # 32

print

# Creating a list of feature column's names

features = df.columns[:4]
# View Features
print features   # to print only column names
'''
Index([u'sepal length (cm)', u'sepal width (cm)', u'petal length (cm)',
       u'petal width (cm)'],
      dtype='object')'''

# converting each species name into digits
y = pd.factorize(train ['species'])[0]  # to convert all T & F into digits
                                        # in order to understand computer

# Viewing target
print y  # length  : 118
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2]'''

# Creating a random forest classifier

clf = RandomForestClassifier(n_jobs= 2,random_state = 0)

clf.fit(train[features],y)

#print test[features]
'''
     sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
7                  5.0               3.4                1.5               0.2
8                  4.4               2.9                1.4               0.2
10                 5.4               3.7                1.5               0.2
13                 4.3               3.0                1.1               0.1
17                 5.1               3.5                1.4               0.3
18                 5.7               3.8                1.7               0.3
19                 5.1               3.8                1.5               0.3
20                 5.4               3.4                1.7               0.2
21                 5.1               3.7                1.5               0.4
23                 5.1               3.3                1.7               0.5
27                 5.2               3.5                1.5               0.2
31                 5.4               3.4                1.5               0.4
38                 4.4               3.0                1.3               0.2'''

# Applying the trained classifier to the test

predction = clf.predict(test[features])
print predction
''' [0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 2 2 1 1 2 2 2 2 2 2 2 2 2 2 2 2]'''


# Viewing the predicted probabilities of the any 10 observations

predictions = clf.predict_proba(test[features])
print predictions[10:26]
'''
[[1.  0.  0. ]
 [1.  0.  0. ]
 [1.  0.  0. ]
 [0.  0.5 0.5]  # if we get equal votes we go for 1st one 
 [0.  1.  0. ]
 [0.  0.9 0.1]
 [0.  0.2 0.8]
 [0.  0.3 0.7]
 [0.  1.  0. ]
 [0.  0.8 0.2]
 [0.  0.  1. ]
 [0.  0.  1. ]
 [0.  0.  1. ]
 [0.  0.  1. ]
 [0.  0.  1. ]
 [0.  0.  1. ]] '''

# Mapping names for the plants for each predicted plant class

pred_plant  = iris.target_names[clf.predict(test[features])]

# view the Predicted trees for the following observations

print pred_plant [:25]   # Forest model predicting trees
'''
['setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'setosa'
 'setosa' 'setosa' 'setosa' 'setosa' 'setosa' 'versicolor' 'versicolor'
 'versicolor' 'virginica' 'virginica' 'versicolor' 'versicolor'
 'virginica' 'virginica' 'virginica' 'virginica' 'virginica']'''

# Vuewing the ACTUAL trees for the 1st 5 observations.

actual_trees =  test['species']
print actual_trees.head()  # here are our actual Trees (input trees)
'''
7     setosa
8     setosa
10    setosa
13    setosa
17    setosa
Name: species, dtype: category
Categories (3, object): [setosa, versicolor, virginica]'''

# Creating Confusion matrix

cm = pd.crosstab(test['species'],pred_plant,rownames = ['Actual Trees'],colnames =['Predicted trees'])

print cm

'''
Predicted trees  setosa  versicolor  virginica
Actual Trees                                  
setosa               13           0          0
versicolor            0           5          2
virginica             0           0         12  '''

# Model Accuracy

print 30.0/32.0*100    # 93.75%

##from sklearn.metrics import confusion_matrix,accuracy_score
##
##Acc_score = clf.score(test['species'],pred_plant)
##
##print  Acc_score 
preds = iris.target_names[clf.predict([[5.1, 3.5, 1.4, 0.2],[6.7, 3.3, 5.7, 2.5],[7.2, 3.6, 6.1, 2.5]])]

print preds  # ['setosa' 'virginica' 'virginica']

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('Iris_flowers_after_prediction')
plt.show()









