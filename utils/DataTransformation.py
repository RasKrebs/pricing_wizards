import pandas as pd
import numpy as np

# Data Preperation for Regression
def base_regression_pipeline(df):
    """
    Prepare data for regression
    """
    
    # Ordinal Encoding for condition, since this typically follows some sort of order
    condition_name = ['Shabby', 'Good but used','Almost as new', 'Never used', 'New, still with price']

    # Replacing rare subsubsub categories with subsubcategories
    minimum = 30 
    rare_sub_categories = pd.DataFrame(df.subsubsubcategory_name.value_counts()).where(df.subsubsubcategory_name.value_counts() < minimum).dropna().index
    df.loc[df[df.subsubsubcategory_name.isin(rare_sub_categories)].index, 'subsubsubcategory_name'] = df[df.subsubsubcategory_name.isin(rare_sub_categories)].subcategory_name
    
    # Mapping brand and subsubsubsub categories average listing prices
    brand_encoding = (df
                      .groupby('brand_name')
                      .agg({'log_listing_price': 'mean'})
                      .sort_values(by='log_listing_price', ascending=True)
                      .to_dict()['log_listing_price'])
    
    subsubsubcategory_encoding = (df
                                  .groupby('subsubsubcategory_name')
                                  .agg({'log_listing_price': 'mean'})
                                  .sort_values(by='log_listing_price', ascending=True)
                                  .to_dict()['log_listing_price'])
    
    # Apply transformations
    df = df.assign(
        condition_name = lambda x: x['condition_name'].apply(lambda x: condition_name.index(x) +1),
        brand_name = lambda x: x['brand_name'].apply(lambda x: brand_encoding[x]),
        subsubsubcategory_name = lambda x: x['subsubsubcategory_name'].apply(lambda x: subsubsubcategory_encoding[x])
    )
    # Final columns to use
    columns_to_use = ['classified_id','log_listing_price','brand_name','condition_name','subsubsubcategory_name']
    df = df[columns_to_use]

    # Return dataframe
    return df

def ridge_regression_pipeline(df):
    # Data transformation for ridge regression
    
    # Minimum count for subsubsubsub category
    minimum = 30
    
    # Mapping rare subsubsubsub categories to subcategory instead
    rare_sub_categories = pd.DataFrame(df.subsubsubcategory_name.value_counts()).where(df.subsubsubcategory_name.value_counts() < minimum).dropna().index
    
    # Replacing with subcategory name
    df.loc[df[df.subsubsubcategory_name.isin(rare_sub_categories)].index, 'subsubsubcategory_name'] = df[df.subsubsubcategory_name.isin(rare_sub_categories)].subsubcategory_name
    
    # Few subsubsubsub categories are still left
    maps = {'Sports shoes': 'Shoes', # Less granular
            'Sportsudstyr': 'Sport', # Less granular
            'Smartphones & Accessories': 'Accessories'} 
    
    # Replacing the few remaining rare cases
    df.loc[df[df.subsubsubcategory_name.isin(maps.keys())].index, 'subsubsubcategory_name'] = df[df.subsubsubcategory_name.isin(maps.keys())].subsubsubcategory_name.map(maps)
    
    # New columns to use
    columns_to_use = ['classified_id', 'log_listing_price', 'brand_name','subsubsubcategory_name']

    # Drop unused columns
    df = df[columns_to_use]

    # OHE columns
    df = pd.get_dummies(df, columns=['brand_name', 'subsubsubcategory_name'])
    
    # Extracting infrequent brands
    infrequent_brands = (df[[col for col in df.columns if 'brand' in col]].sum(axis=0).sort_values(ascending=True) < 50)
    infrequent_brands = infrequent_brands[infrequent_brands == True].index

    # Assigning 'other' to brands that are in infrequent_brands
    df['brand_name_other'] = df[infrequent_brands].sum(axis=1)
    df = df.drop(columns=infrequent_brands)

    return df
