# Estimating Home Values from Zillow

### Table of Contents
- 1.) Project Overview
- 2.) Project Description
- 3.) Project Goals
- 4.) Project Deliverables
<br>
<br>

 - 5.) Project Summary
 - 6.) Hypothesis
 - 7.) Findings and Next Steps
<br>
<br>

 - 8a.) Planning
 - 8b.) Data Acquisition
 - 8c.) Data Preparation
 - 8d.) Data Exploration
 - 8e.) Modeling and Evaluation
 - 8f.) Delivery/Deliverables
 <br>
 <br>

 - 9.) Conclusion and Next Steps 
 - 10.) Data Dictionary
 - 11.) How to Recreate


 ## 1.) Project Overview
 In this project, I will be working with a Zillow dataset to create a model that will predict a property's value. Specifically for this scenario...
 This machine learning model will predict single-unit property values that are sold during 2017. 
 <br>
 <br>

## 2.) Project Description
  We are to build a regression model to predict a continuous variable (tax_value) using features that will help us better predict property value. Since there are heaps of missing data, we'll have to find ways to handle it. We also want to know which counties these properties are located in and calculate tax rate. 
    
 ## 3.) Goals
  - Deliver a Jupyter notebook going through the steps of the data science pipeline
  - Create a regression model that performs better than the baseline
  - Present to audience about my findings
 <br>
 <br>

## 4.) Deliverables
 - Finalized Jupyter notebook complete with comments
 - A README.md with executive summary, contents, data dictionary, conclusion and next steps, and how to recreate this project.
 <br>
 <br>

## 5.) Project Summary
   I built a regression model to predict single unit property values in 3 California counties using a Zillow data frame. 

## 6.) Hypothesis

1.) The larger the square footage, the higher the property value

2.) The more bedrooms or bathrooms a house has, the higher its property value will be

3.) The older a house is, the less it will be worth.

4.) Value is dependent on property location

## 7.)  Findings and Next Steps
   - Square footage was the best feature for predicting home value, followed up by bathrooms and bedrooms.
   - Age may have had a little factor, so it was accepted to use to better our model, but there may be other features we could look into next time.
   - Location still may have a factor in value, but we would need data that is more normally distributed. Most of the properties were in Los Angeles County. 

Next steps would be:
 - gather more information on location
 - try to clean up/fill in missing values for other location-based columns such as ZIP code, longitude/latitude
 - clean up other columns on home features and see if our model would perform with them (lower RMSE, higher r^2) 

<br>
<br>

# 8.) The Pipeline: 

## 8a.) Planning 
Goal: Plan out the project
I will be seeing how square footage, bathroom count, and bedroom count relate to property value. I believe there will be a 
positive correlation among these variables. 

I also want to look into other features, like age and FIPS code, and see if that will also correlate to property value. 
A lot of these features could play hand in hand and help my model make better predictions.

Hypotheses: Square footage, number of bedrooms, number of bathrooms have a positive relationship with value. Age has a negative relationship with value. FIPS codes have an affect on value, perhaps the means are different across each county. 


<br>

## 8b.) Acquire 
Goal: Have Zillow dataframe ready to prepare in acquire.py
In this stage, I used a connection URL to access the CodeUp database. Using a SQL query, I brought in the Zillow dataset with only properties set for single use, and were sold in 2017. I turned it into a pandas dataframe and created a .csv in order to use it for the rest of the pipeline. 
| acquire.py Functions | Purpose                                                        |
|----------------------|----------------------------------------------------------------|
| get_connection()     | Creates a connection link so we can access our data            |
| new_zillow_data()    | Uses a SQL query to return Zillow into a data frame            |
| get_zillow_data()    | Returns Zillow as a data frame, and creates a local Zillow.csv |
<br>


For the next stage: Drop or fill in nulls, remove outliers, rename columns, make new relevant columns.

<br>

## 8c.) Prep 
Goal: Have Zillow dataset that is split into train, validate, test, and ready to be analyzed. Assure data types are appropriate and that missing values/duplicates/outliers are addressed. Put this in a prep.py. 
In this stage, I handled outliers by dropping any rows with values that were 3 standard deviations above or below the mean.
I assured that all columns had a numeric data type, and renamed them for ease of use.
Duplicates were dropped (in parcelid)
Nulls were also dropped, due to the strong correlation between square feet in respect to property value. I did not want to risk making the model dependent on assumed values. 
I split the data into train, validate, test, X_train, y_train, X_validate, y_validate, X_test, and y_test.
Last, I scaled it on a min-max scaler (I made sure to drop outliers first!) and also returned X_train, X_validate, and X_test scaled. 
<br>
| prep.py                                                  | Purpose                                                           |
|----------------------------------------------------------|-------------------------------------------------------------------|
| remove_outlier(df)                                       | removes outliers for certain columns                              |
| clean_zillow(df)                                         | obtain certain columns, makes it ready for use                    |
| train_validate_test(df, target)                          | X sets and y sets                                                 |
| get_object_cols(df)                                      | returns columns with object data types                            |
| get_numeric_X_cols(df, object_cols)                      | returns columns with numeric data types                           |
| min_max_scale(X_train, X_validate, X_test, numeric_cols) | uses MinMax scaler on X sets                                      |
<br>

For the next step: run statistical testing and visualize data to find relationships between variables.
<br>


## 8d.) Explore 
Goal: Visualize the data. Explore relationships.  Find answers. Use the visuals and statistics tests to help answer your questions. 
I plotted distributions, made sure nothing was out of the ordinary after cleaning the dataset. 

Plotted a pairplot to see combinations of variables.

I ran a few t-tests with the features in respect to tax_value. Also a few to see if the independent variables were related to each other. 

I found that square footage, bedroom count, and bathroom count were all statistically significant. They are not independent to property value. Bedroom count and bathroom count were also dependent on each other. 
<br>

| explore.py Functions                               | Purpose                                                                  |
|----------------------------------------------------|--------------------------------------------------------------------------|
| plot_variable_pairs(train, cols, hue=None)         | displays pairplot with regression line                                   |
| plot_pairplot(train, cols, hue=None)               | displays pairplot with scatter plots and histograms                      |
| correlation_exploration(train, x_string, y_string) | shows visual correlation between two columns                             |
| get_zillow_heatmap(train)                          | returns a heat map and r values of how each feature relates to tax_value |

<br>
For the next step: Select features to use to build a regression model that predicts property value
<br>

## 8e.) Modeling and Evaluation 
Goal: develop a regression model that performs better than the baseline.

The models worked best with sqft, baths, beds, and age. Polynomial Regression performed the best, so I did a test on it.

| Model                            | RMSE Training | RMSE Validate | R^2   |
|----------------------------------|---------------|---------------|-------|
| Baseline                         | 413268.163083 | 413818.472866 | -0.000122|
| LinearRegression                 | 324177.799545 | 321334.985074 | 0.396955 |
| LassoLars                        | 324178.208202 | 321348.607211 | 0.396904 |
| TweedieRegressor                 | 324177.799545 | 321334.985069 | 0.396955 |
| PolynomialRegression (3 degrees) | 318899.434281 | 316565.929491 | 0.414723 |
<br>

Test:
 - RMSE of 323,772.490847
 - R^2 of 0.4006944135449

Beats the baseline! 
<br>
## 8f.) Delivery
I will be giving a presentation over my findings!
 - All acquire, prepare, wrangle, model, explore .py files are uploaded for easy replication.
 - This README 
 - Final notebook that documents a commented walkthrough of my process

## 9.) Conclusion

To conclude...
We took a very large Zillow dataset and condensed it down to 52,200 rows to work with. We dropped rows with outliers of 3 standard deviations above or below the mean for that column. 

 -  Square footage was the best feature to determine a property's value. As square footage increased, it seemed that value also went up.

 -  The more bedrooms and bathrooms a house has, the more it was worth. These number of rooms also related to square footage in a positive relationship.

 -  Age was not a huge factor in value, but was helpful in our model's predictions.

 -  Using all of square footage, number of bedrooms, number of bathrooms, and age into a model performed better than the baseline. 



## 10.) Data Dictionary 

| Column Name                  | Renamed          | Info                                            | Data type      |
|------------------------------|------------------|-------------------------------------------------|----------------|
| parcelid                     | N/A              | ID of the property (unique)                     | Object          |
| bathroomcnt                  | number_bathroom  | number of bathrooms                             | Float          |
| bedroomcnt                   | number_bedroom   | number of bedrooms                              | Float          |
| calculatedfinishedsquarefeet | sqft             | number of square feet                           | Float          |  
| fips                         | geographic_code  | FIPS code (for county)                          | Int            |
| yearbuilt                    | N/A              | The year the property was built                 | Float          |   
| taxvaluedollarcnt            | tax_value        | Property's tax value in dollars                 | Float          |
| age                          | N/A              | 2017-yearbuilt (to see the age of the property) | Float          | 
| county                       | N/A              | 6037-Los Angeles, 6059-Orange, 6111-Ventura     | Object
 

<br>
<br>

## 11.) How to Recreate Project

 - You'll need your own username/pass/host credentials in order to use the get_connection function in my acquire.py to access the Zillow database
 - Have a copy of my acquire, prep, explore .py files. You can adjust the features to use, how to handle outliers, etc. or just keep it the way I have it! 
 - My final notebook has all of the steps outlined, and it is really easy to adjust parameters.