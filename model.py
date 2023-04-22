import pandas as pd
import numpy as np

import sklearn.metrics as metric
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

def regression_models(X_train, y_train, X_validate, y_validate):
    '''
    Takes in X_train, y_train, X_validate, y_validate and runs 
    different models and produces df with RMSE and r^2 scores
    for each model on train and validate.
    '''
    
    train_predictions = pd.DataFrame(y_train)
    validate_predictions = pd.DataFrame(y_validate)

    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame() 

    #OLS Model
    lm = LinearRegression(normalize=True)
    lm.fit(X_train, y_train)
    train_predictions['lm'] = lm.predict(X_train)
    # predict validate
    validate_predictions['lm'] = lm.predict(X_validate)
    metric_df = make_metric_df(y_train, train_predictions.lm, y_validate, validate_predictions.lm, metric_df, model_name = 'OLS Regressor')

    #Lasso Lars
    # create the model object
    lars = LassoLars(alpha=1)
    lars.fit(X_train, y_train)
    # predict train
    train_predictions['lars'] = lars.predict(X_train)
    # predict validate
    validate_predictions['lars'] = lars.predict(X_validate)
    metric_df = make_metric_df(y_train, train_predictions.lars, y_validate, validate_predictions.lars, metric_df, model_name = 'Lasso_alpha_1')
    
    #Tweedie Regressor/GLM
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)
    glm.fit(X_train, y_train)
    # predict train
    train_predictions['glm'] = glm.predict(X_train)
    # predict validate
    validate_predictions['glm'] = glm.predict(X_validate)
    metric_df = make_metric_df(y_train, train_predictions.glm, y_validate, validate_predictions.glm, metric_df, model_name = 'GLM')

    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    # create the model object
    lm2 = LinearRegression(normalize=True)
    lm2.fit(X_train_degree2, y_train)
    # predict train
    train_predictions['poly_2'] = lm2.predict(X_train_degree2)
    # predict validate
    validate_predictions['poly_2'] = lm2.predict(X_validate_degree2)
    metric_df = make_metric_df(y_train, train_predictions.poly_2, y_validate, validate_predictions.poly_2, metric_df, model_name = 'Quadratic')

    return metric_df

def make_metric_df(y_train, y_train_pred, y_validate, y_validate_pred,  metric_df,model_name ):
    '''
    Takes in y_train, y_train_pred, y_validate, y_validate_pred, and a df
    returns a df of RMSE and r^2 score for the model on train and validate
    '''
    if metric_df.size ==0:
        metric_df = pd.DataFrame(data=[
            {
                'model': model_name, 
                f'RMSE_train': metric.mean_squared_error(
                    y_train,
                    y_train_pred) ** .5,
                f'r^2_train': metric.explained_variance_score(
                    y_train,
                    y_train_pred),
                f'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    y_validate_pred) ** .5,
                f'r^2_validate': metric.explained_variance_score(
                    y_validate,
                    y_validate_pred)
            }])
        return metric_df
    else:
        return metric_df.append(
            {
                'model': model_name, 
                f'RMSE_train': metric.mean_squared_error(
                    y_train,
                    y_train_pred) ** .5,
                f'r^2_train': metric.explained_variance_score(
                    y_train,
                    y_train_pred),
                f'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    y_validate_pred) ** .5,
                f'r^2_validate': metric.explained_variance_score(
                    y_validate,
                    y_validate_pred)
            }, ignore_index=True)

def baseline_models(y_train, y_validate):
    '''
    Takes in y_train and y_validate and returns a df of 
    baseline_mean and baseline_median and how they perform
    '''
    train_predictions = pd.DataFrame(y_train)
    validate_predictions = pd.DataFrame(y_validate)
    
    y_pred_mean = y_train.mean()
    train_predictions['y_pred_mean'] = y_pred_mean
    validate_predictions['y_pred_mean'] = y_pred_mean
    
    y_pred_median = y_train.median()
    train_predictions['y_pred_median'] = y_pred_median
    validate_predictions['y_pred_median'] = y_pred_median

    # create the metric_df as a blank dataframe
    metric_df = pd.DataFrame(data=[
    {
        'model': 'mean_baseline', 
        'RMSE_train': metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5,
        'RMSE_validate': metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5,
        'Difference': (( metric.mean_squared_error(
            y_train,
            train_predictions['y_pred_mean']) ** .5)-(metric.mean_squared_error(
            y_validate,
            validate_predictions['y_pred_mean']) ** .5))
    }])

    return metric_df.append(
            {
                'model': 'median_baseline', 
                'RMSE_train': metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5,
                'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5,
                'Difference': (( metric.mean_squared_error(
                    y_train,
                    train_predictions['y_pred_median']) ** .5)-(metric.mean_squared_error(
                    y_validate,
                    validate_predictions['y_pred_median']) ** .5))
            }, ignore_index=True)

def best_model(X_train, y_train, X_validate, y_validate, X_test, y_test):
    '''
    Takes in X_train, y_train, X_validate, y_validate, X_test, y_test
    and returns a df with the RMSE and r^2 score on train, validate and test
    '''    
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2)
    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train)
    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate)
    X_test_degree2 = pf.transform(X_test)
    # create the model object
    lm2 = LinearRegression(normalize=True)
    lm2.fit(X_train_degree2, y_train)
    
    metric_df = pd.DataFrame(data=[
            {
                'model': 'Quadratic', 
                f'RMSE_train': metric.mean_squared_error(
                    y_train,
                    lm2.predict(X_train_degree2)) ** .5,
                f'r^2_train': metric.explained_variance_score(
                    y_train,
                    lm2.predict(X_train_degree2)),
                f'RMSE_validate': metric.mean_squared_error(
                    y_validate,
                    lm2.predict(X_validate_degree2)) ** .5,
                f'r^2_validate': metric.explained_variance_score(
                    y_validate,
                    lm2.predict(X_validate_degree2)),
                f'RMSE_test': metric.mean_squared_error(
                    y_test,
                    lm2.predict(X_test_degree2)) ** .5,
                f'r^2_test': metric.explained_variance_score(
                    y_test,
                    lm2.predict(X_test_degree2))
            }])
    
    return metric_df