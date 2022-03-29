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
    WHERE propertylandusedesc IN ("Single Family Residential")
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
    
    