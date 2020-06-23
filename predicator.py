# Import all needed libraries
import pandas as pd # to manipulate dataframes
import numpy as np # for computation
from sklearn.model_selection import train_test_split # to split data between train and test data
from sklearn.linear_model import LinearRegression # import linear regression model 
from sklearn.neighbors import KNeighborsRegressor # import knn model
from sklearn.metrics import r2_score # import metric to test model
import datetime # to manipulate datetime data types 

def check_data(df):
    '''
    Description:
    Remove rows with null values

    INPUT :
        - df : dataframe to clean
    
    OUTPUT :
        - df : dataframe without null values
    '''
    null = df[df.isnull().sum(axis = 1) > 0]
    df.drop(index=null.index, axis = 1, inplace = True)

    return df

def set_params(df, start, end):
    '''
    Description :
    Select data within the training range date from the whole data

    INPUT :
        - df : dataframe with the whole data
        - start : 1st date of the range
        - end : last date of the range

    OUTPUT :
        - df : dataframe of the trainin data 
    '''
    df['Date'] = pd.to_datetime(df.Date)
    df = df[df.Date < pd.to_datetime(end)]
    df = df[df.Date > pd.to_datetime(start)]

    return df

def prediction_preparation(X):
    '''
    Description :
    Convert company values into dummy columns and append them to the dataframe

    INPUT :
        - X : dataframe to dummy
    
    OUTPUT :
        - X : dataframe with dummy columns
    '''
    X = pd.concat([X, pd.get_dummies(X.Company)], axis = 1)
    X.drop(columns='Company', axis = 1, inplace = True)

    return X

def linear_regression(X, Y, test_size, random_state = 42):
    '''
    Description :
    Create a linear regression model

    INPUT :
        - X : dataframe of the parameters to make the predictions
        - Y : dataframe of the values to predict
        - test_size : the size of the test/train dataframes
    OUTPUT :
        - model : linear regression model
        - Y_preds : predicted values
        - Y_test : test values

    '''
    X = prediction_preparation(X)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    
    model = LinearRegression(normalize=False)
    model.fit(X_train, Y_train)
    Y_preds = model.predict(X_test)

    return model, Y_preds, Y_test

def knn(X, Y, test_size, random_state = 42):
    '''
    Description :
    Create a knn model

    INPUT :
        - X : dataframe of the parameters to make the predictions
        - Y : dataframe of the values to predict
        - test_size : the size of the test/train dataframes
    OUTPUT :
        - model : linear regression model
        - Y_preds : predicted values
        - Y_test : test values

    '''
    
    k = np.min([np.round(np.log(test_size)+1, 0).astype('int'), 3])

    X = prediction_preparation(X)
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size / X.shape[0], random_state=random_state)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size / X.shape[0], random_state=random_state)

    model = KNeighborsRegressor(n_neighbors=k, weights='distance')
    model.fit(X_train, Y_train)
    Y_preds = model.predict(X_test)
        
    return model, Y_preds, Y_test

def bagging(df, forecasted_date):
    '''
    Description :
    60 random bags are created from the original data. For each bag, a model is built :
        - a LR model if the forecast is <= 2
        - a knn model if the forecast is > 2
    The score is computed by comparing the average of the testing values and the predicted values of all bags
    The bagging technique is used to have a better prediction performace, as well as the use of the LR model 
    for a forecast <= 2 and the knn model for a higher forecast

    INPUT :
        - df : dataframe of the data to predict
        - forecasted_date : forecast due date
    
    OUTPUT :
        - bags : bags of models
        - score : average bags score
    '''
    bags = {}
    preds = []
    tests = []

    #Create bags of LR and knn models
    for i in range(60):

        # select random rows for a bag
        df_sub = df.sample(np.round(df.shape[0], 0).astype('int'), replace = True)
        df_sub = df_sub.sort_values(by='Date')

        # split in X / Y datasets
        X = df_sub.drop(columns = ['Adj Close', 'Date'])
        Y = df_sub['Adj Close']

        # Create bag model
        if forecasted_date <= 2:
            model, Y_preds, Y_test = linear_regression(X, Y, forecasted_date)
        else:
            model, Y_preds, Y_test = knn(X, Y, forecasted_date)
        
        # append bag model
        bags[i] = model
        preds.append(Y_preds)
        tests.append(Y_test.values)
    df_preds = pd.DataFrame(preds)
    df_tests = pd.DataFrame(tests)

    global_preds = df_preds.mean(axis=0)
    global_tests = df_tests.mean(axis=0)

    # score
    score = r2_score(global_preds, global_tests)
    if len(global_preds) == 1:
        if global_preds[0] > global_tests.iloc[0]:
            score = global_tests.iloc[0] / global_preds[0]
        else:
            score = global_preds[0] / global_tests.iloc[0]
    
    return bags, score

def main(df, today, start_train, forecasted_date):
    '''
    Description:
    Prepare the data for the training and train models

    INPUT :
        - df : dataframe of the original data
        - today : today's date
        - start_train : period for trainig the models
        - forecast_date : period to forecast

    OUTPUT :
        - bags : model bags
        - score : average score

    '''
    # Remove rows with missing values
    df = check_data(df.drop(columns=['Data']))
    
    #Shift the values to predict
    sub2 = df['Adj Close'][forecasted_date:]
    sub1 = df.drop(columns = ['Adj Close'])[:-forecasted_date]
    df_sub = pd.concat([sub1.reset_index(drop=True), sub2.reset_index(drop=True)], axis=1)

    # Set the training range
    df_train = set_params(df_sub, start_train, today)

    # bagging
    bags, score = bagging(df_train, forecasted_date)

    return bags, score 

