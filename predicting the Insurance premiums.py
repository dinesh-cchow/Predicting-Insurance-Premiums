#!/usr/bin/env python
# coding: utf-8

# # Predicting Insurance Premiums
# ~ our dataset contains a Few attributes for each person such as Age,sex,BMI,children,smocker,region and their charges
# 
# Target
# 
# ~ To use this info to predict charges for new customers

# In[7]:


import pandas as pd


# In[8]:


# Load the insurance data 
dt=pd.read_csv(r'D:\data science and deepl learning 20 case studies\datascienceforbusiness-master\insurance.csv')


# In[9]:


# Lets check our data or preview 
dt.head()


# In[11]:


# Let's see the type and info about our data
dt.info()


# In[12]:


# Let's check the statstics of our data for the features 
dt.describe()


# In[20]:


# Summary os shape of data,features in data,Missing values in data,Unique values in data
print('(rows,columns):',dt.shape)
print('\nfeatures:\n',dt.columns.tolist())
print('missing values:',dt.isnull().sum())
print('unique vales:',dt.nunique())


# In[21]:


# correlation matrix to watch out in the data 
dt.corr()


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


#plot to check the correlation with more insight 

def plot_corr(df,size=10):
    corr=df.corr()
    fig, ax=plt.subplots(figsize=(size,size))
    ax.legend()
    cax=ax.matshow(corr)
    fig.colorbar(cax)
    plt.xticks(range(len(corr.columns)),corr.columns,rotation='vertical')
    plt.yticks(range(len(corr.columns)),corr.columns)
    
plot_corr(dt)


# # Exploratory Data Analaysis

# In[26]:


# Lets do some Exploratory data analysis for the insights in the data 
fig,axes=plt.subplots(nrows=2,ncols=2,figsize=(12,10))
dt.plot(kind='hist',y='age',bins=70,color='b',ax=axes[0][0])
dt.plot(kind='hist',y='bmi',bins=100,color='r',ax=axes[0][1])
dt.plot(kind='hist',y='children',bins=6,color='g',ax=axes[1][0])
dt.plot(kind='hist',y='charges',bins=100,color='orange',ax=axes[1][1])
plt.show()


# In[27]:


dt['sex'].value_counts().plot(kind='bar')


# In[29]:


dt['smoker'].value_counts().plot(kind='bar')


# In[33]:



fig,axes=plt.subplots(nrows=1,ncols=3,figsize=(15,5))
dt.plot(kind='scatter',x='age',y='charges',alpha=0.5,color='green',ax=axes[0],title='Age vs Charges')
dt.plot(kind='scatter',x='bmi',y='charges',alpha=0.5,color='blue',ax=axes[1],title='Bmi vs Charges')
dt.plot(kind='scatter',x='children',y='charges',alpha=0.5,color='red',ax=axes[2],title='Children vs Charges')

plt.show()


# In[34]:


#import seaborn library
import seaborn as sns


# In[35]:


pal=['#FA5858','#58D3F7']
sns.scatterplot(x='bmi',y='charges',data=dt,palette=pal,hue='smoker')


# In[36]:


# Violin plot
pal=['#FA5858','#58D3F7']
sns.catplot(x='sex',y='charges',data=dt,palette=pal,kind='violin',hue='smoker')


# In[37]:


# pair plot for the data
sns.set(style='ticks')
pal=['#FA5858','#58D3F7']

sns.pairplot(dt,hue='smoker',palette=pal)
plt.title('smokers')


# # Preparing data for Machine Learning Algorithms

# In[38]:


dt.head()


# In[40]:


dt['region'].unique()


# In[41]:


#Lets remove the region
dt.drop(['region'],axis=1,inplace=True)
dt.head()


# In[42]:


# changing the binary categories to 1s and 0s
dt.sex.replace({'female':1,'male':0},inplace=True)
dt.smoker.replace({'yes':1,'no':0},inplace=True)


# In[43]:


#Let's check whether the categories have been changed to os and 1s or not
dt.head()


# In[44]:


# separating features and target variable from the data
x=dt.drop(['charges'],axis=1)
y=dt.charges


# # modeling of our data

# In[49]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

model=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=101)

model.fit(x_train,y_train)

y_train_pred=model.predict(x_train)
y_test_pred=model.predict(x_test)

print(model.score(x_test,y_test))


# Score is the R2 score, which varies between 0 and 100%,it is closely related to MSE but not the same
# it is the proportion of the variance in the dependent variable that is predictable form the independent
# (Totalvariance expained by model)/Total variance. if it is 100% then two variable are perfectly correlated

# In[50]:


results=pd.DataFrame({'actual':y_test,'predicted':y_test_pred})
results


# In[53]:


# Let's normalize the data
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[67]:


#Lets Train Linear regression model on the normalized data

multiple_linear_reg=LinearRegression(fit_intercept=False)

multiple_linear_reg.fit(x_train,y_train)


# In[61]:


from sklearn.preprocessing import PolynomialFeatures

polynomial_features=PolynomialFeatures(degree=3)  # create a polynomialfeatures instance in degree3

x_train_poly=polynomial_features.fit_transform(x_train)  #fit and tranform the training data to polynomial
x_test_poly=polynomial_features.fit_transform(x_test)    #fit and transform the testing dta to polynomial

polynomial_reg=LinearRegression(fit_intercept=False)  # create a instance for Linear Regression model

polynomial_reg.fit(x_train_poly,y_train)     #fit data to the model


# In[62]:


from sklearn.tree import DecisionTreeRegressor   #import decision tree regressor

decision_tree_reg=DecisionTreeRegressor(max_depth=5,random_state=101) # create a instance for decision tree regressor
decision_tree_reg.fit(x_train,y_train)   # fit data to the model


# In[63]:


from sklearn.ensemble import RandomForestRegressor  #import Random Fores regressor

random_forest_reg=RandomForestRegressor(n_estimators=400,max_depth=5,random_state=101) # create a instance for decision tree regressor
random_forest_reg.fit(x_train,y_train)   # fit data to the model


# In[65]:


from sklearn.svm import SVR #import support vector regressor

support_vector_reg=SVR(gamma='auto',kernel='linear',C=1000)  #create a instance for support vector regression model
support_vector_reg.fit(x_train,y_train)   #fir data to the model


# In[66]:


from sklearn.model_selection import cross_val_predict # for k-fold cross validation
from sklearn.metrics import r2_score  # for finding the accuracy with R2 score
from sklearn.metrics import mean_squared_error #for MSE
from math import sqrt  # for square root operation


# # Evaluating Multiple Linear Regression Model

# In[69]:


# prediction with training data set
y_pred_mlr_train=multiple_linear_reg.predict(x_train)

#prediction with the testing dataset
y_pred_mlr_test=multiple_linear_reg.predict(x_test)

#find training accuracy for this model:
accuracy_mlr_train=r2_score(y_train,y_pred_mlr_train)
print('Training accuracy for Multiple Linear Regression Model:',accuracy_mlr_train)
# find testing accuracy for this model:

accuracy_mlr_test=r2_score(y_test,y_pred_mlr_test)
print('Testing accuracy for Multiple Linear Regression Model:',accuracy_mlr_test)

# Find RMSE for training data
rmse_mlr_train=sqrt(mean_squared_error(y_train,y_pred_mlr_train))
print('rmse for training data:',rmse_mlr_train)

# Find RMSE for testing data
rmse_mlr_test=sqrt(mean_squared_error(y_test,y_pred_mlr_test))
print('rmse for training data:',rmse_mlr_train)

# prediction with 10-fold cross validation

y_pred_cv_mlr=cross_val_predict(multiple_linear_reg,x,y,cv=10)

#finding accuracy after 10 fold cross validation

accuracy_cv_mlr=r2_score(y,y_pred_cv_mlr)
print('Accuracy for 10-fold cross validation multiple linear regression model:',accuracy_cv_mlr)


# # Evaluating polynomial regression model

# In[71]:


# Prediction with training dataset:
y_pred_PR_train = polynomial_reg.predict(x_train_poly)

# Prediction with testing dataset:
y_pred_PR_test = polynomial_reg.predict(x_test_poly)

# Find training accuracy for this model:
accuracy_PR_train = r2_score(y_train, y_pred_PR_train)
print("Training Accuracy for Polynomial Regression Model: ", accuracy_PR_train)

# Find testing accuracy for this model:
accuracy_PR_test = r2_score(y_test, y_pred_PR_test)
print("Testing Accuracy for Polynomial Regression Model: ", accuracy_PR_test)

# Find RMSE for training data:
RMSE_PR_train = sqrt(mean_squared_error(y_train, y_pred_PR_train))
print("RMSE for Training Data: ", RMSE_PR_train)

# Find RMSE for testing data:
RMSE_PR_test = sqrt(mean_squared_error(y_test, y_pred_PR_test))
print("RMSE for Testing Data: ", RMSE_PR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_PR = cross_val_predict(polynomial_reg, polynomial_features.fit_transform(x), y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_PR = r2_score(y, y_pred_cv_PR)
print("Accuracy for 10-Fold Cross Predicted Polynomial Regression Model: ", accuracy_cv_PR)


# # Evaluating Decision Tree Regression Model

# In[75]:


# Prediction with training dataset:
y_pred_DTR_train = decision_tree_reg.predict(x_train)

# Prediction with testing dataset:
y_pred_DTR_test = decision_tree_reg.predict(x_test)

# Find training accuracy for this model:
accuracy_DTR_train = r2_score(y_train, y_pred_DTR_train)
print("Training Accuracy for Decision Tree Regression Model: ", accuracy_DTR_train)

# Find testing accuracy for this model:
accuracy_DTR_test = r2_score(y_test, y_pred_DTR_test)
print("Testing Accuracy for Decision Tree Regression Model: ", accuracy_DTR_test)

# Find RMSE for training data:
RMSE_DTR_train = sqrt(mean_squared_error(y_train, y_pred_DTR_train))
print("RMSE for Training Data: ", RMSE_DTR_train)

# Find RMSE for testing data:
RMSE_DTR_test = sqrt(mean_squared_error(y_test, y_pred_DTR_test))
print("RMSE for Testing Data: ", RMSE_DTR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_DTR = cross_val_predict(decision_tree_reg, x, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_DTR = r2_score(y, y_pred_cv_DTR)
print("Accuracy for 10-Fold Cross Predicted Decision Tree Regression Model: ", accuracy_cv_DTR)


# # Evaluating Random Forest Regression Model

# In[77]:


# Prediction with training dataset:
y_pred_RFR_train = random_forest_reg.predict(x_train)

# Prediction with testing dataset:
y_pred_RFR_test = random_forest_reg.predict(x_test)

# Find training accuracy for this model:
accuracy_RFR_train = r2_score(y_train, y_pred_RFR_train)
print("Training Accuracy for Random Forest Regression Model: ", accuracy_RFR_train)

# Find testing accuracy for this model:
accuracy_RFR_test = r2_score(y_test, y_pred_RFR_test)
print("Testing Accuracy for Random Forest Regression Model: ", accuracy_RFR_test)

# Find RMSE for training data:
RMSE_RFR_train = sqrt(mean_squared_error(y_train, y_pred_RFR_train))
print("RMSE for Training Data: ", RMSE_RFR_train)

# Find RMSE for testing data:
RMSE_RFR_test = sqrt(mean_squared_error(y_test, y_pred_RFR_test))
print("RMSE for Testing Data: ", RMSE_RFR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_RFR = cross_val_predict(random_forest_reg, x, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_RFR = r2_score(y, y_pred_cv_RFR)
print("Accuracy for 10-Fold Cross Predicted Random Forest Regression Model: ", accuracy_cv_RFR)


# # Evaluating Support Vector Regression Model

# In[78]:


# Prediction with training dataset:
y_pred_SVR_train = support_vector_reg.predict(x_train)

# Prediction with testing dataset:
y_pred_SVR_test = support_vector_reg.predict(x_test)

# Find training accuracy for this model:
accuracy_SVR_train = r2_score(y_train, y_pred_SVR_train)
print("Training Accuracy for Support Vector Regression Model: ", accuracy_SVR_train)

# Find testing accuracy for this model:
accuracy_SVR_test = r2_score(y_test, y_pred_SVR_test)
print("Testing Accuracy for Support Vector Regression Model: ", accuracy_SVR_test)

# Find RMSE for training data:
RMSE_SVR_train = sqrt(mean_squared_error(y_train, y_pred_SVR_train))
print("RMSE for Training Data: ", RMSE_SVR_train)

# Find RMSE for testing data:
RMSE_SVR_test = sqrt(mean_squared_error(y_test, y_pred_SVR_test))
print("RMSE for Testing Data: ", RMSE_SVR_test)

# Prediction with 10-Fold Cross Validation:
y_pred_cv_SVR = cross_val_predict(support_vector_reg, x, y, cv=10)

# Find accuracy after 10-Fold Cross Validation
accuracy_cv_SVR = r2_score(y, y_pred_cv_SVR)
print("Accuracy for 10-Fold Cross Predicted Support Vector Regression Model: ", accuracy_cv_SVR)


# In[82]:


# Compare all results in one table
training_accuracies = [accuracy_mlr_train, accuracy_PR_train, accuracy_DTR_train, accuracy_RFR_train, accuracy_SVR_train]
testing_accuracies = [accuracy_mlr_test, accuracy_PR_test, accuracy_DTR_test, accuracy_RFR_test, accuracy_SVR_test]
training_RMSE = [rmse_mlr_train, RMSE_PR_train, RMSE_DTR_train, RMSE_RFR_train, RMSE_SVR_train]
testing_RMSE = [rmse_mlr_test, RMSE_PR_test, RMSE_DTR_test, RMSE_RFR_test, RMSE_SVR_test]
cv_accuracies = [accuracy_cv_mlr, accuracy_cv_PR, accuracy_cv_DTR, accuracy_cv_RFR, accuracy_cv_SVR]

parameters = ["fit_intercept=False", "fit_intercept=False", "max_depth=5", "n_estimators=400, max_depth=5", "kernel=”linear”, C=1000"]

table_data = {"Parameters": parameters, "Training Accuracy": training_accuracies, "Testing Accuracy": testing_accuracies, 
              "Training RMSE": training_RMSE, "Testing RMSE": testing_RMSE, "10-Fold Score": cv_accuracies}
model_names = ["Multiple Linear Regression", "Polynomial Regression", "Decision Tree Regression", "Random Forest Regression", "Support Vector Regression"]

table_dataframe = pd.DataFrame(data=table_data, index=model_names)
table_dataframe


# # Our best classifier is our Random Forests using 400 estimators and a max_depth of 5

# **R^2 (coefficient of determination) regression score function.**
# 
# Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

# # Let's test our best regression on some new data

# In[84]:


# The data is of mine let's check how much premium i will be paying from the model which we have built
input_data = {'age': [23],
              'sex': ['male'],
              'bmi': [13],
              'children': [0],
              'smoker': ['yes'],
              'region': ['southeast']}

input_data = pd.DataFrame(input_data)
input_data


# In[85]:


# Our simple pre-processing 
input_data.drop(["region"], axis=1, inplace=True) 
input_data['sex'] = input_data['sex'].map(lambda s :1  if s == 'female' else 0)
input_data['smoker'] = input_data['smoker'].map(lambda s :1  if s == 'yes' else 0)
input_data


# In[86]:


# Scale our input data  
input_data = sc.transform(input_data)
input_data


# In[87]:


# Reshape our input data in the format required by sklearn models
input_data = input_data.reshape(1, -1)
print(input_data.shape)
input_data


# In[88]:


# Get our predicted insurance rate for our new customer
random_forest_reg.predict(input_data)


# In[89]:


# Note Standard Scaler remembers your inputs so you can use it still here
print(sc.mean_)
print(sc.scale_)


# In[ ]:




