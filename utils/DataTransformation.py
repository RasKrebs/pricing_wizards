import pandas as pd
import numpy as np

# Data Preperation for Regression
def base_regression_pipeline(df):
    """
    Prepare data for regression
    """
    
    # Condition Mapping
    condition_name = ['Shabby', 'Good but used','Almost as new', 'Never used', 'New, still with price']
    
    # Brand Encoding
    brand_encoding = df.groupby('brand_name').agg({'log_listing_price': 'mean'}).sort_values(by='log_listing_price', ascending=True).to_dict()['log_listing_price']
    
    # Mapping rare subsubsubsub categories to subcategory instead
    minimum = 100 # Minimum number of listings
    rare_sub_categories = pd.DataFrame(df.subsubsubcategory_name.value_counts()).where(df.subsubsubcategory_name.value_counts() < minimum).dropna().index
    
    # Mapping rare subsubsubsub categories a mean price
    subsubsubcategory_encoding = df.groupby('subsubsubcategory_name').agg({'log_listing_price': 'mean'}).sort_values(by='log_listing_price', ascending=True).to_dict()['log_listing_price']
    
    # Apply transformations
    df = df.assign(
        condition_name = lambda x: x['condition_name'].apply(lambda x: condition_name.index(x) +1),
        brand_name = lambda x: x['brand_name'].apply(lambda x: brand_encoding[x] + 1)
    )
    df.loc[df[df.subsubsubcategory_name.isin(rare_sub_categories)].index, 'subsubsubcategory_name'] = df[df.subsubsubcategory_name.isin(rare_sub_categories)].subcategory_name
    df['subsubsubcategory_name'] = df['subsubsubcategory_name'].apply(lambda x: subsubsubcategory_encoding[x])
    
    # Final columns to use
    columns_to_use = ['classified_id','log_listing_price','log_viewed_count','brand_name','condition_name','subsubsubcategory_name']
    df = df[columns_to_use]
    
    # Return dataframe
    return df