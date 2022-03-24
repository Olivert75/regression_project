import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression


def remove_outlier(df):
    '''
    This function will remove values that are 3 standard deviations above or below the mean for sqft, baths, beds, and tax_value.         (Our MVP values)
    '''
    new_df = df[(np.abs(stats.zscore(df['sqft'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['baths'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['beds'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['tax_value'])) < 3)]
    return new_df

def clean_zillow (df):
    '''
    Takes in a df and drops duplicates, nulls,rename columns, parcelid is changed to string.
    Return a clean df
    '''
    
    # drop duplicates
    df = df.drop_duplicates()
    #drop nulls
    df = df.dropna(how='any',axis=0)
    #  change parcelid to string, it is a uniqure identifier for parcels lots
    df['parcelid'] = df['parcelid'].astype('str')
    df['fips'] = df['fips'].astype('int')
    # rename columns
    df = df.rename(columns={'parcelid':'parcel_id',
                            'calculatedfinishedsquarefeet': 'sqft',
                            'bathroomcnt': 'number_bathroom',
                            'bedroomcnt': 'number_bedroom',
                            'taxvaluedollarcnt':'tax_value',
                            'taxamount': 'tax_amount',
                            'fips':'geographic_code'})

    return df

def clean_zillow_data(df):
    '''
    this function takes in an unclean zillow df and does the following:
    1.) keeps only columns we need are considering. 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt', 'taxvaluedollarcnt', 'yearbuilt','fips'
    2.) drops nulls
    3.) renames columns for ease of use.
    4.) creates new columns that we may use.
    '''
    #select features for df
    features = ['parcelid','calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt', 'taxvaluedollarcnt', 'yearbuilt','fips', 'taxamount']
    df = df[features]
    #  change parcelid to string, it is a uniqure identifier for parcels lots
    df['parcelid'] = df['parcelid'].astype('str')
    df['fips'] = df['fips'].astype('int')
    #for the yearbuilt column, fill in nulls with 2017.
    df['yearbuilt'].fillna(2017, inplace = True)
    #create a new column named 'age', which is 2017 minus the yearbuilt
    df['age'] = 2017-df['yearbuilt']
    #calculate tax_rate by having taxamount divided by taxvaluedollarcnt
    df['tax_rate'] = df['taxamount'] / df['taxvaluedollarcnt']
    #drop duplicates in parcelid
    df = df.drop_duplicates(subset=['parcelid'])
    #rename columns for easier use
    df = df.rename(columns={'parcelid':'parcel_id',
                            'calculatedfinishedsquarefeet': 'sqft',
                            'bathroomcnt': 'number_bathroom',
                            'bedroomcnt': 'number_bedroom',
                            'taxvaluedollarcnt':'tax_value',
                            'taxamount': 'tax_amount',
                            'fips':'geographic_code'
        
    })
    
    #set index
    df = df.set_index('parcel_id')
    #drop nulls in sqft and tax_value
    df = df.dropna(subset=['sqft','tax_value','tax_amount'])
    #drop year_built, we can just use age.
    df = df.drop(columns=['yearbuilt'])
    
    return df
    
def split_data(df):
    '''
    take in a DataFrame and return train, validate, and test DataFrames.
    
    '''
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, 
                                       test_size=.3, 
                                       random_state=123)
    print(f'train -> {train.shape}')
    print(f'validate -> {validate.shape}')
    print(f'test -> {test.shape}')                                  
    return train, validate, test

def split_Xy (train, validate, test, target):
    '''
    This function takes in three dataframe (train, validate, test) and a target  and splits each of the 3 samples
    into a dataframe with independent variables and a series with the dependent, or target variable.
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test.
    Example:
    X_train, y_train, X_validate, y_validate, X_test, y_test = split_Xy (train, validate, test, 'Fertility' )
    '''
    
    #split train
    X_train = train.drop(columns= [target])
    y_train= train[target]
    #split validate
    X_validate = validate.drop(columns= [target])
    y_validate= validate[target]
    #split validate
    X_test = test.drop(columns= [target])
    y_test= test[target]

    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 
    return  X_train, y_train, X_validate, y_validate, X_test, y_test

def train_validate_test(df, target):
    '''
    this function combine from 2 functions above, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. Then print the shape of 3 dataframes (Train, Validate and Test)
    The function returns train, validate, test sets and also another 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

        
    # split train into X (dataframe, drop target) & y (series, keep target only)
    X_train = train.drop(columns=[target])
    y_train = train[target]
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    X_validate = validate.drop(columns=[target])
    y_validate = validate[target]
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    X_test = test.drop(columns=[target])
    y_test = test[target]
    
    print(f'X_train -> {X_train.shape}               y_train->{y_train.shape}')
    print(f'X_validate -> {X_validate.shape}         y_validate->{y_validate.shape} ')        
    print(f'X_test -> {X_test.shape}                  y_test>{y_test.shape}') 

    return train, validate, test, X_train, y_train, X_validate, y_validate, X_test, y_test


def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def get_numeric_X_cols(X_train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in X_train.columns.values if col not in object_cols]
    
    return numeric_cols

def min_max_scale(X_train, X_validate, X_test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).


    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    X_train_scaled_array = scaler.transform(X_train[numeric_cols])
    X_validate_scaled_array = scaler.transform(X_validate[numeric_cols])
    X_test_scaled_array = scaler.transform(X_test[numeric_cols])

    # convert arrays to dataframes
    X_train_scaled = pd.DataFrame(X_train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([X_test.index.values])

    
    return X_train_scaled, X_validate_scaled, X_test_scaled

def scaled_df ( train_df , validate_df, test_df, scaler):
    '''
    Take in a 3 df and a type of scaler that you  want to  use. it will scale all columns
    except object type. Fit a scaler only in train and tramnsform in train, validate and test.
    returns  new dfs with the scaled columns.
    scaler : MinMaxScaler() or RobustScaler(), StandardScaler() 
    Example:
    scaled_df( X_train , X_validate , X_test, RobustScaler())
    
    '''
    #get all columns except object type
    columns = train_df.select_dtypes(exclude='object').columns.tolist()
    
    # fit our scaler
    scaler.fit(train_df[columns])
    # get our scaled arrays
    train_scaled = scaler.transform(train_df[columns])
    validate_scaled= scaler.transform(validate_df[columns])
    test_scaled= scaler.transform(test_df[columns])

    # convert arrays to dataframes
    train_scaled_df = pd.DataFrame(train_scaled, columns=columns).set_index([train_df.index.values])
    validate_scaled_df = pd.DataFrame(validate_scaled, columns=columns).set_index([validate_df.index.values])
    test_scaled_df = pd.DataFrame(test_scaled, columns=columns).set_index([test_df.index.values])

    #plot
    for col in columns: 
        plt.figure(figsize=(13, 6))
        plt.subplot(121)
        plt.hist(train_df[col], ec='black')
        plt.title('Original')
        plt.xlabel(col)
        plt.ylabel("counts")
        plt.subplot(122)
        plt.hist(train_scaled_df[col],  ec='black')
        plt.title('Scaled')
        plt.xlabel(col)
        plt.ylabel("counts")

    return train_scaled_df, validate_scaled_df, test_scaled_df
    
def unique_cntvalues (df, max_unique):
    '''
    takes in a df and a max_unique values to show using value_counts().
    returns a report of number of unique values foe each columns and shouw unique values that < max_unique
    '''
    #checking the uniques values for each column
    columns = df.columns.tolist()
    print( '************************** COUNT OF UNIQUE VALUES ************************** ')
    print( 'Columns')
    print(" ")
    cat_list = []
    for col in columns:
        print(f'{col} --> {df[col].nunique()} unique values')
        if df[col].nunique() < max_unique:
            cat_list.append(col)
        print(" ")
    #checking  the variables that have few values
    print(" **************************  UNIQUE VALUES **************************")
    print(" ")
    print(f"Uniques values of all the columns that have less than {max_unique} unique values ")
    print(" ")
    for l in cat_list:
        print(l)
        print(df[l].value_counts().sort_index())
        print("--------------------------- ")
        print(" ")


def remove_outlier_tax(df):
    '''
    Another outlier removal function. This one will remove values that are 3 standard deviations above or below the mean for our MVP columns, and for our tax_value, tax_amounts.
    '''
    new_df = df[(np.abs(stats.zscore(df['sqft'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['baths'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['beds'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['tax_value'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['tax_amount'])) < 3)]
    return new_df

# plot distributions
def distribution (df):
    '''
    takes in a df and plot individual variable distributions excluding object type
    '''
    cols =df.columns.to_list()
    for col in cols:
        if df[col].dtype != 'object':
            plt.hist(df[col])
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.ylabel('Counts of customers')
            plt.show()

def distribution_boxplot (df):
    '''
    takes in a df and boxplot variable distributions excluding object type
    '''
    cols =df.columns.to_list()
    for col in cols:
        if df[col].dtype != 'object':
            plt.figure(figsize=(8, 6))
            sns.boxplot(x= col, data=df)
            plt.title(f'Distribution of {col}')
            plt.xlabel('values')
            plt.show()
    return
