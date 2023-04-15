#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


# Importing Library
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Reading the training dataset in a dataframe using Pandas
df = pd.read_csv("train.csv")

# Reading the test dataset in a dataframe using Pandas
test = pd.read_csv("test.csv")


# In[2]:


# First 10 Rows of training Dataset

df.head(10)


# In[3]:


# Store total number of observation in training dataset
df_length =len(df)

# Store total number of columns in testing data set
test_col = len(test.columns)


# # Understanding the various features (columns) of the dataset.

# In[4]:


# Summary of numerical variables for training data set

df.describe()


# 1. For the non-numerical values (e.g. Property_Area, Credit_History etc.), we can look at frequency distribution to understand whether they make sense or not.

# In[5]:


# Get the unique values and their frequency of variable Property_Area

df['Property_Area'].value_counts()


# 

# In[6]:


# Box Plot for understanding the distributions and to observe the outliers.

get_ipython().run_line_magic('matplotlib', 'inline')

# Histogram of variable ApplicantIncome

df['ApplicantIncome'].hist()


# In[7]:


# Box Plot for variable ApplicantIncome of training data set

df.boxplot(column='ApplicantIncome')


# 3. The above Box Plot confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. 

# In[8]:


# Box Plot for variable ApplicantIncome by variable Education of training data set

df.boxplot(column='ApplicantIncome', by = 'Education')


# 4. We can see that there is no substantial different between the mean income of graduate and non-graduates. But there are a higher number of graduates with very high incomes, which are appearing to be the outliers

# In[9]:


# Histogram of variable LoanAmount

df['LoanAmount'].hist(bins=50)


# In[10]:


# Box Plot for variable LoanAmount of training data set

df.boxplot(column='LoanAmount')


# In[11]:


# Box Plot for variable LoanAmount by variable Gender of training data set

df.boxplot(column='LoanAmount', by = 'Gender')


# 5. LoanAmount has missing as well as extreme values, while ApplicantIncome has a few extreme values.

# In[12]:


# Loan approval rates in absolute numbers
loan_approval = df['Loan_Status'].value_counts()['Y']
print(loan_approval)


# - 422 number of loans were approved.

# In[13]:


# Credit History and Loan Status
pd.crosstab(df ['Credit_History'], df ['Loan_Status'], margins=True)


# In[14]:


#Function to output percentage row wise in a cross table
def percentageConvert(ser):
    return ser/float(ser[-1])

# # Loan approval rate for customers having Credit_History (1)
#df['Y'] = pd.crosstab(df ["Credit_History"], df ["Loan_Status"], margins=True).apply(percentageConvert, axis=1)
#loan_approval_with_Credit_1 = df['Y'][1]
#print(loan_approval_with_Credit_1*100)


# - 79.58 % of the applicants whose loans were approved have Credit_History equals to 1.

# In[15]:


df.head()


# In[16]:


# Replace missing value of Self_Employed with more frequent category
df['Self_Employed'].fillna('No',inplace=True)


# In[17]:


# Add both ApplicantIncome and CoapplicantIncome to TotalIncome
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']

# Looking at the distribtion of TotalIncome
df['LoanAmount'].hist(bins=20)


# - The extreme values are practically possible, i.e. some people might apply for high value loans due to specific needs. So instead of treating them as outliers, let’s try a log transformation to nullify their effect:

# In[18]:


# Perform log transformation of TotalIncome to make it closer to normal
df['LoanAmount_log'] = np.log(df['LoanAmount'])

# Looking at the distribtion of TotalIncome_log
df['LoanAmount_log'].hist(bins=20)


# 

# In[19]:


# Impute missing values for Gender
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)

# Impute missing values for Married
df['Married'].fillna(df['Married'].mode()[0],inplace=True)

# Impute missing values for Dependents
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

# Impute missing values for Credit_History
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

# Convert all non-numeric values to number
cat=['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']

for var in cat:
    le = preprocessing.LabelEncoder()
    df[var]=le.fit_transform(df[var].astype('str'))
df.dtypes


# In[30]:


from sklearn.model_selection import KFold

def classification_model(model, data, predictors, outcome):
    # Fit the model
    model.fit(data[predictors], data[outcome])
  
    # Make predictions on training set
    predictions = model.predict(data[predictors])
  
    # Print accuracy
    accuracy = metrics.accuracy_score(predictions, data[outcome])
    print("Accuracy: %s" % "{0:.3%}".format(accuracy))

    # Perform k-fold cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    error = []
    for train_index, test_index in kf.split(data):
        # Split data into training and testing sets
        train_predictors = data[predictors].iloc[train_index]
        train_target = data[outcome].iloc[train_index]
        test_predictors = data[predictors].iloc[test_index]
        test_target = data[outcome].iloc[test_index]

        # Train the model on training data
        model.fit(train_predictors, train_target)
    
        # Record error from each cross-validation run
        error.append(model.score(test_predictors, test_target))
 
    print("Cross-Validation Score: %s" % "{0:.3%}".format(np.mean(error)))

    # Fit the model again so that it can be referred outside the function
    model.fit(data[predictors], data[outcome])


# # Model Building

# In[21]:


#Combining both train and test dataset

#Create a flag for Train and Test Data set
df['Type']='Train' 
test['Type']='Test'
fullData = pd.concat([df,test], axis=0)

#Look at the available missing values in the dataset
fullData.isnull().sum()


# In[22]:


#Identify categorical and continuous variables
ID_col = ['Loan_ID']
target_col = ["Loan_Status"]
cat_cols = ['Credit_History','Dependents','Gender','Married','Education','Property_Area','Self_Employed']


# In[23]:


#Imputing Missing values with mean for continuous variable
fullData['LoanAmount'].fillna(fullData['LoanAmount'].mean(), inplace=True)
fullData['LoanAmount_log'].fillna(fullData['LoanAmount_log'].mean(), inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mean(), inplace=True)
fullData['ApplicantIncome'].fillna(fullData['ApplicantIncome'].mean(), inplace=True)
fullData['CoapplicantIncome'].fillna(fullData['CoapplicantIncome'].mean(), inplace=True)

#Imputing Missing values with mode for categorical variables
fullData['Gender'].fillna(fullData['Gender'].mode()[0], inplace=True)
fullData['Married'].fillna(fullData['Married'].mode()[0], inplace=True)
fullData['Dependents'].fillna(fullData['Dependents'].mode()[0], inplace=True)
fullData['Loan_Amount_Term'].fillna(fullData['Loan_Amount_Term'].mode()[0], inplace=True)
fullData['Credit_History'].fillna(fullData['Credit_History'].mode()[0], inplace=True)


# In[24]:


#Create a new column as Total Income

fullData['TotalIncome']=fullData['ApplicantIncome'] + fullData['CoapplicantIncome']

fullData['TotalIncome_log'] = np.log(fullData['TotalIncome'])

#Histogram for Total Income
fullData['TotalIncome_log'].hist(bins=20) 


# In[25]:


#create label encoders for categorical features
for var in cat_cols:
    number = LabelEncoder()
    fullData[var] = number.fit_transform(fullData[var].astype('str'))

train_modified=fullData[fullData['Type']=='Train']
test_modified=fullData[fullData['Type']=='Test']
train_modified["Loan_Status"] = number.fit_transform(train_modified["Loan_Status"].astype('str'))


# In[ ]:





# In[26]:


from sklearn.linear_model import LogisticRegression


predictors_Logistic=['Credit_History','Education','Gender']

x_train = train_modified[list(predictors_Logistic)].values
y_train = train_modified["Loan_Status"].values

x_test=test_modified[list(predictors_Logistic)].values


# In[31]:


# Create logistic regression object
model = LogisticRegression()

# Train the model using the training sets
model.fit(x_train, y_train)

#Predict Output
predicted= model.predict(x_test)

#Reverse encoding for predicted outcome
predicted = number.inverse_transform(predicted)

#Store it to test dataset
test_modified['Loan_Status']=predicted

outcome_var = 'Loan_Status'

classification_model(model, df,predictors_Logistic,outcome_var)

test_modified.to_csv("Logistic_Prediction.csv",columns=['Loan_ID','Loan_Status'])


# In[ ]:




