import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split



def remove_outlier(df):
    '''
    This function will remove values that are 3 standard deviations above or below the mean for sqft, baths, beds, and tax_value.
    '''
    new_df = df[(np.abs(stats.zscore(df['sqft'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['number_bathroom'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['number_bedroom'])) < 3)]
    new_df = df[(np.abs(stats.zscore(df['tax_value'])) < 3)]
    return new_df

def clean_zillow (df):
    '''
    this function takes in an unclean zillow df and does the following:
    1.) keeps only columns we need are considering. 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt', 'taxvaluedollarcnt', 'yearbuilt','fips'
    2.) drops nulls
    3.) renames columns for ease of use.
    4.) creates new columns that we may use.
    return a clean df
    '''
    
    # drop duplicates
    df = df.drop_duplicates()
    #drop nulls
    df = df.dropna(how='any',axis=0)
    #  change parcelid to string, it is a uniqure identifier for parcels lots
    df['parcelid'] = df['parcelid'].astype('str')
    df['fips'] = df['fips'].astype('int')
    #create a new column named 'age', which is 2017 minus the yearbuilt
    df['age'] = 2017-df['yearbuilt']
    #create a new column named 'county', map it with the code in fips
    df['county'] = df['fips'].map({6037: 'Los Angeles', 6059: 'Orange', 6111: 'Ventura'})
    # rename columns
    df = df.rename(columns={'parcelid':'parcel_id',
                            'calculatedfinishedsquarefeet': 'sqft',
                            'bathroomcnt': 'number_bathroom',
                            'bedroomcnt': 'number_bedroom',
                            'taxvaluedollarcnt':'tax_value',
                            'fips':'geographic_code'})
    #drop year_built, we can just use age.
    df = df.drop(columns=['yearbuilt'])

    return df

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
