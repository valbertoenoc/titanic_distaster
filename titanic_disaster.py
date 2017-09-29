# data manipulation
import pandas as pd
import numpy as np

# data analysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer

# data visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as snso

# string manipulation
import re

# load raw data
titanic_train = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')

# selecting general features of interest
print(titanic_test.describe())
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch', 'Name']

data_train = titanic_train[features].copy()
data_test = titanic_test[features].copy()
print(data_train.shape)

def dataSummary(data):
	print(data.describe(), '\n', '-'*40)
	print(data.info(), '\n', '-'*40)
	print(data.shape, '\n', '-'*40)
	print(data.head(), '\n', '-'*40)
	print(data.tail(), '\n', '-'*40)

# fill not available values in age with median age
def fillMissingValues(data):
    data.loc[:, 'Age'] = data.Age.fillna(data.Age.median())
    data.loc[:, 'Fare'] = data.Fare.fillna(data.Fare.median()) 
    data.loc[:, 'Embarked'] = data.Embarked.fillna('S')
    
# convert features to numericals
def convertFeatures(data):
    data.loc[:, 'Pclass'] = data.loc[:, 'Pclass'].astype('int')
    data.loc[:, 'Sex'] = data.Sex.map({'male': 0, 'female': 1})
    data.loc[:, 'Embarked'] = data.Embarked.map({'S':0, 'C': 1, 'Q': 2})
    
    ''' Categorize Age '''
    data.loc[(data['Age'] <= 20), 'Age'] = 0
    data.loc[(data['Age'] > 20) & (data['Age'] <= 28), 'Age'] = 1
    data.loc[(data['Age'] > 28) & (data['Age'] <= 38), 'Age'] = 2
    data.loc[(data['Age'] > 38) & (data['Age'] <= 80), 'Age'] = 3
    data.loc[(data['Age'] > 80)] = 4

    ''' Categorize Fare '''
    data.loc[(data['Fare'] <= 7.91), 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.343) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[(data['Fare'] > 31), 'Fare'] = 3
    
    ''' Convert data type '''
    data.loc[:, 'Age'] = data.Age.astype('int')
    data.loc[:, 'Fare'] = data.Fare.astype('int')

def getTitle(data):
	data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

	data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
	data['Title'] = data['Title'].replace('Mlle', 'Miss')
	data['Title'] = data['Title'].replace('Ms', 'Miss')
	data['Title'] = data['Title'].replace('Mme', 'Mrs')

	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
	data['Title'] = data['Title'].map(title_mapping)


# adds family size feature
def addFamilyFeature(data):
	data['FamilySize'] = data.SibSp + data.Parch + 1
    
fillMissingValues(data_train)
fillMissingValues(data_test)
convertFeatures(data_train)
convertFeatures(data_test)
addFamilyFeature(data_train)
addFamilyFeature(data_test)
getTitle(data_train)
getTitle(data_test)

data_train = data_train.drop('Name', axis=1)
data_test = data_test.drop('Name', axis=1)

# dataSummary(data_train)

def perform_logistic_regression(train_X, train_Y, test_X):
	lr = LogisticRegression()
	lr.fit(train_X, train_Y)
	pred_y = lr.predict(test_X)
	accuracy = round(lr.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'Logistic Regression', 'Accuracy': accuracy}
	return returnval

def perform_decision_tree(train_X, train_Y, test_X):
	dt = DecisionTreeClassifier()
	dt.fit(train_X, train_Y)
	pred_y = dt.predict(test_X)
	accuracy = round(dt.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'Decision Tree', 'Accuracy': accuracy}
	return returnval

def perform_svc(train_X, train_Y, test_X):
	svc_clf = SVC()
	svc_clf.fit(train_X, train_Y)
	svc_clf.predict(test_X)
	accuracy = round(svc_clf.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'SVC', 'Accuracy': accuracy}
	return returnval


def perform_linear_svc(train_X, train_Y, test_X):
	svc_linear_clf = LinearSVC()
	svc_linear_clf.fit(train_X, train_Y)
	svc_linear_clf.predict(test_X)
	accuracy = round(svc_linear_clf.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'Linear SVC', 'Accuracy': accuracy}
	return returnval

def perform_rfc(train_X, train_Y, test_X):
	rfc = RandomForestClassifier(n_estimators=100, oob_score=True, max_features=None)
	rfc.fit(train_X, train_Y)
	rfc.predict(test_X)
	accuracy = round(rfc.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'Random Forest Classifier', 'Accuracy': accuracy}
	return returnval

def perform_knn(train_X, train_Y, test_X):
	knn = KNeighborsClassifier()
	knn.fit(train_X, train_Y)
	knn.predict(test_X)
	accuracy = round(knn.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'KNN Classifiers', 'Accuracy': accuracy}
	return returnval

def perform_gnb(train_X, train_Y, test_X):
	gnb = GaussianNB()
	gnb.fit(train_X, train_Y)
	gnb.predict(test_X)
	accuracy = round(gnb.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'Gaussian NB Classifier', 'Accuracy': accuracy}
	return returnval

def perform_xgb(train_X, train_Y, test_X):
	# xgboost parameters
	parameters = { 'learning_rate' : [0.1],
	               'n_estimators' : [40],
	               'max_depth': [3],
	               'min_child_weight': [3],
	               'gamma':[0.4],
	               'subsample' : [0.8],
	               'colsample_bytree' : [0.8],
	               'scale_pos_weight' : [1],
	               'reg_alpha':[1e-5]
	             } 

	clf = XGBClassifier() 
	grid_obj = GridSearchCV(clf,
	                        # scoring=xgb_val,
	                        param_grid=parameters,
	                        cv=5)
	grid_obj = grid_obj.fit(train_x, train_y)
	clf = grid_obj.best_estimator_
	print(clf)

	# xgb = XGBClassifier()
	clf.fit(train_X, train_Y)
	clf.predict(test_X)
	accuracy = round(clf.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'XGBoost Classifier', 'Accuracy': accuracy}
	return returnval

def perform_mlp(train_X, train_Y, test_X):
	mlp = MLPClassifier(hidden_layer_sizes=(10), learning_rate_init=0.01, max_iter=500)
	mlp.fit(train_X, train_Y)
	mlp.predict(test_X)
	accuracy = round(mlp.score(train_X, train_Y) * 100, 2)
	returnval = {'Model': 'MLP Classifier', 'Accuracy': accuracy}
	return returnval

train_x = data_train
train_y = titanic_train['Survived']
test_x = data_test.copy() 

lg_val = perform_logistic_regression(train_x, train_y, test_x)
dt_val = perform_decision_tree(train_x, train_y, test_x)
svc_val = perform_svc(train_x, train_y, test_x)
linear_svc_val = perform_linear_svc(train_x, train_y, test_x)
gnb_val = perform_gnb(train_x, train_y, test_x)
knn_val = perform_knn(train_x, train_y, test_x)
rfc_val = perform_rfc(train_x, train_y, test_x)
xgb_val = perform_xgb(train_x, train_y, test_x)
mlp_val = perform_mlp(train_x, train_y, test_x)

models = [lg_val, dt_val, svc_val, linear_svc_val, gnb_val, knn_val, rfc_val, xgb_val, mlp_val]

model_accuracies = pd.DataFrame(models)
cols = list(model_accuracies.columns.values)
cols = cols[-1:] + cols[:-1]
model_accuracies = model_accuracies[cols]
model_accuracies = model_accuracies.sort_values('Accuracy')

print(model_accuracies) 

# xgboost parameters
parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             } 

clf = XGBClassifier() 
grid_obj = GridSearchCV(clf,
                        # scoring=xgb_val,
                        param_grid=parameters,
                        cv=5)
grid_obj = grid_obj.fit(train_x, train_y)
clf = grid_obj.best_estimator_
print(clf)

def create_submission(clf, train_X, train_Y, test_X, test_data):
	# clf = RandomForestClassifier(n_estimators=100, oob_score=True, max_features=None)
	clf.fit(train_X, train_Y)
	pred_y = clf.predict(test_X)
	pred_y_list = pred_y.tolist()
	test_X['Survived'] = pred_y
	test_X['PassengerId'] = test_data['PassengerId']

	submission_data = test_X[['PassengerId', 'Survived']]
	submission_data.to_csv('passenger_survived.csv', sep=',', index=False)

create_submission(clf, train_x, train_y, test_x, titanic_test)

