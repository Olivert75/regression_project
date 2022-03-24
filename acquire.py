import os
import pandas as pd
from env import user_name, host, password 

def get_connection(db):
    '''
    Creates a connection URL
    '''
    return f'mysql+pymysql://{user_name}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    Returns zillow into a dataframe
    '''
    sql_query =''' 
    SELECT  parcelid,
            bedroomcnt, 
            bathroomcnt, 
            calculatedfinishedsquarefeet, 
            taxvaluedollarcnt, 
            yearbuilt, 
            fips,
            taxamount 
    FROM properties_2017
    JOIN propertylandusetype USING (propertylandusetypeid)
    JOIN predictions_2017 as pred USING(parcelid)
    WHERE propertylandusedesc IN ("Single Family Residential", "Inferred Single Family Residential")
    AND transactiondate LIKE "2017%%";
    '''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df 

def get_zillow_data():
    '''get connection, returns zillow into a dataframe and creates a csv for us'''
    if os.path.isfile('zillow.csv'):
        df = pd.read_csv('zillow.csv', index_col=0)
    else:
        df = new_zillow_data()
        df.to_csv('zillow.csv')
    return df
    
    
    
    
def clean_zillow(df):
    '''
    this function takes in an unclean zillow df and does the following:
    1.) keeps only columns we need for the project
    2.) drops nulls
    3.) renames columns
    '''
    #select features for df, took these features from my acquire exercise
    features = ['parcelid', 'calculatedfinishedsquarefeet', 'bathroomcnt', 'bedroomcnt', 'taxvaluedollarcnt','yearbuilt','taxamount','fips']
    df = df[features]

    
    #rename columns for easier use
    df = df.rename(columns={
                            'parcelid': 'parcel_id',
                            'calculatedfinishedsquarefeet': 'sqft',
                            'bathroomcnt': 'baths',
                            'bedroomcnt': 'beds',
                            'taxvaluedollarcnt':'tax_value',
                            'yearbuilt':'year_built',
                            'taxamount': 'tax_amount'
        
    })
    
    #set index
    df = df.set_index('parcel_id')
    #drop nulls
    df = df.dropna(subset=['sqft','tax_value'])
    
    return df
    
    