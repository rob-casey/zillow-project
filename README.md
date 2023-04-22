# Zillow Project
## An educational ML Project using Regression to predict home values

## Project Goals
* Identify key drivers of value for single family properties.
* Build a Machine Learning Regression model that effectively predicts property tax assessed values using the identified drivers.
* Review findings and provide recommendations for future improvements of the model's performance.

## The Plan
### Acquire the data from Codeup mySQL database

## Data dictionary
| Feature | Definition | Type |
|:--------|:-----------|:-------
|**parcelid**| Definition|*int*|
|**bathroooms**| The number of bathrooms in the home |*float*|
|**bedrooms**| The number of bedrooms in the home |*int*|
|**square_feet**| Square footage of the house |*int*|
|**year_built**| Year the house was built |*int*|
|**bath_bed_ratio**| The number of bathrooms divided by number of bedrooms |*float*|
|**county**| Name of the county where the house is located |*string*|
|**2017_age**| Age of the house in 2017 (when the data was collected |*int*|
|**home_value**| The tax-assessed value of the home |*float*|

### Prepare data
#### Dropped rows:
* Duplicates   
* Rows having 0 bedrooms AND 0 bathrooms 
* Rows having more than 10,000 square feet
* Rows containing null values in any column

#### Created features
* ```county``` (names based on the fips code):  
    - 6037: LA
    - 6059: Orange 
    - 6111: Ventura 
* ```bath_bed_ratio``` 
    - Column displaying bathrooms/bedrooms

#### Other prep
* Split data into train, validate, and test sets

### Explore data in search of drivers of churn
* Answer the following initial question
    1. Is there a significant relationship between square footage and home value?
    2. Is there a significant relationship between the bath-to-bed ratio and home value? 
    3. Does location have a relationship with home value?
    4. Is there a significant relationship between age of the home and home value?

### Develop a model to predict the value of a house
* Use drivers identified through exploration to build different predictive models
* Evaluate models on train and validate data
* Select best model based on highest $r^2 score$
* Evaluate the best model on the test data

## Conclusion

### Summary
* ```square_feet``` appears to be a driver of home value
* ```bath_bed_ratio``` appears to be a driver of home value
* ```county``` appears to be a driver of home value
* ```2017_age``` appears to be a driver of home value


### Recommendations
* Run this model against other fips clusters to evaluate it's performance in states and cities other than California

### Next Steps
* If new trends are discovered between features while evaluating the model's performance in other fips clusters, engineer new ones
* Then rerun the newly engineered features with the Orange-Ventura-LA county data for any possible relationship with home value
