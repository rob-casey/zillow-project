import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

def square_feet_viz(train):
    '''
    Takes in the train data and returns a lmplot comparing square_feet
    and home_value
    '''
    #set font size
    sns.set(font_scale=1.5)
    #set plot style
    sns.set_style('white')
    
    #make plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    sns.regplot('square_feet', 'home_value', data=train, scatter_kws={"color":"#0173b2"}, line_kws={"color": "#de8f05"})
    ax.set_title("Square Footage Compared with Home Value")
    ax.set_xlabel("Square Feet")
    ax.set_ylabel("Home Value (millions of dollars)")
    plt.show()

def sq_feet_home_distribution(train):
    '''
    Takes in the train data and returns a histplot of the 
    distribution of houses by square footage
    '''
    #set plot style
    sns.set_style('white')
    
    #make plot
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    sns.histplot(data = train, x = 'square_feet', kde=True, color="#0173b2", ax=ax)
    ax.set_title('Distribution of Homes by Square Footage')
    ax.set_xlabel('Square Feet')
    ax.set_ylabel('Number of Homes')
    plt.show()

def sq_feet_spearmanr(train):
    '''
    Takes in the train data and runs a spearman's R test on
    square feet and home value, returns the correlation coefficient
    and p-value
    '''
    #run spearman test to see if there is any linear correlation
    corr, p = stats.spearmanr(train.square_feet, train.home_value)

    #print results
    print(f"Correlation Coefficient: {corr:.2}\np-value: {p:.4}")

def bath_bed_viz(train):
    '''
    Takes in the train data and returns a lmplot comparing bath_bed_ratio
    and home_value
    '''
    #set font size
    sns.set(font_scale=1.5)
    #set plot style
    sns.set_style('white')
    
    #make plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    sns.regplot('bath_bed_ratio', 'home_value', data=train, scatter_kws={"color":"#0173b2"}, line_kws={"color": "#de8f05"})
    ax.set_title("Bath/Bed Ratio Compared with Home Value")
    ax.set_xlabel("Bath/Bed Ratio")
    ax.set_ylabel("Home Value (millions of dollars)")
    plt.show()

def bath_bed_ratio_spearmanr(train):
    '''
    Takes in the train data and runs a spearman's R test on
    bath_bed_ratio and home value, returns the correlation 
    coefficient and p-value
    '''
    #run spearman test to see if there is any linear correlation
    corr, p = stats.spearmanr(train.bath_bed_ratio, train.home_value)

    #print results
    print(f"Correlation Coefficient: {corr:.2}\np-value: {p:.4}")

def county_viz(train):
    '''
    Takes in train data and creates a box plot for the three counties
    by home value
    '''
    #make plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    sns.boxplot(data = train, x = 'county', y = 'home_value', palette='colorblind', ax=ax)
    ax.set_title('Home Value by County')
    ax.set_xlabel('County')
    ax.set_ylabel('Home Value (millions of dollars)')
    plt.show()

def county_ANOVA(train):
    '''
    Takes in train data, splits by county and runs an ANOVA test
    returns f score and p-value
    '''
    #splitting data into a df per county:
    la = train[train.county == 'LA'].home_value
    orange = train[train.county == 'Orange'].home_value
    ventura = train[train.county == 'Ventura'].home_value

    #Running a one-way ANOVA (there is a single independent variable - 'county'):
    f, p = stats.f_oneway(la, orange, ventura)

    #print results
    print(f"F-score: {f:.8}\np-value: {p:.4}")

def age_viz(train):
    '''
    Takes in the train data and returns a lmplot comparing 2017_age
    and home_value
    '''
    #set font size
    sns.set(font_scale=1.5)
    #set plot style
    sns.set_style('white')
    
    #make plot
    fig, ax = plt.subplots(1,1, figsize=(10,8))
    sns.regplot('2017_age', 'home_value', data=train, scatter_kws={"color":"#0173b2"}, line_kws={"color": "#de8f05"})
    ax.set_title("Age in 2017 Compared with Home Value")
    ax.set_xlabel("Age in 2017")
    ax.set_ylabel("Home Value (millions of dollars)")
    plt.show()

def age_distribution(train):
    '''
    Takes in train data and returns a histplot of the
    distribution of homes by 2017_age
    '''
    #set plot style
    sns.set_style('white')
    
    #make plot
    fig, ax = plt.subplots(1,1, figsize=(8,5))
    sns.histplot(data = train,  x = '2017_age', kde=True, color="#0173b2", ax=ax)
    ax.set_title('Distribution of Homes by 2017 Age')
    ax.set_xlabel('Age in 2017')
    ax.set_ylabel('Number of Homes')
    plt.show()

def age_spearmanr(train):
    '''
    Takes in the train data and runs a Pearsons's R test on
    2017_age and home value, returns the correlation 
    coefficient and p-value
    '''
    #run spearman test to see if there is any linear correlation
    corr, p = stats.spearmanr(train['2017_age'], train.home_value)

    #print results
    print(f"Correlation Coefficient: {corr:.8}\np-value: {p:.4}")