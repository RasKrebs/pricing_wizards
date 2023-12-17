import pandas as pd
import numpy as np

def condition_encoding(df):
    """
    Ordinal encoding for condition
    """
    # Condition name ordered
    condition_name = ['Shabby', 'Good but used','Almost as new', 'Never used', 'New, still with price']
    
    # Apply ordinal encoding
    df.condition_name = df.condition_name.apply(lambda x: condition_name.index(x) +1)   
    
    return df


def price_encoding(df, column, listing_price_column='log_listing_price'):
    """
    Ordinal encoding for brand
    """
    brand_encoding = (df
                      .groupby(column)
                      .agg({listing_price_column: 'mean'})
                      .sort_values(by=listing_price_column, ascending=True)
                      .to_dict()[listing_price_column])
    
    df[column] = df[column].apply(lambda x: brand_encoding[x])
    
    return df


def filter_rare_categories(df, col, replacement, minimum=30):
    """
    Filter rare subsubsub categories
    """
    # Set minimum
    minimum = minimum
    
    # Extract rare categories
    rare_categories = pd.DataFrame(df[col].value_counts()).where(df[col].value_counts() < minimum).dropna().index
    
    # Replace
    df.loc[df[df[col].isin(rare_categories)].index, 'subsubsubcategory_name'] = df[df[col].isin(rare_categories)][replacement]
    
    # Few subsubsubsub categories remain - these are mapped to subcategory instead
    maps = {'Sports shoes': 'Shoes', # Less granular
            'Sportsudstyr': 'Sport', # Less granular
            'Smartphones & Accessories': 'Accessories'} 
    
    # Replacing the few remaining rare cases
    df.loc[df[df.subsubsubcategory_name.isin(maps.keys())].index, 'subsubsubcategory_name'] = df[df.subsubsubcategory_name.isin(maps.keys())].subsubsubcategory_name.map(maps)
    
    return df


# Data Preperation for Regression
def base_regression_pipeline(df):
    """
    Prepare data for regression
    """

    # Replacing rare subsubsub categories with subsubcategories
    df = filter_rare_categories(df, 'subsubsubcategory_name', 'subsubcategory_name')
    df = condition_encoding(df)
    df = price_encoding(df, 'brand_name')
    df = price_encoding(df, 'subsubsubcategory_name')
    
    # Final columns to use
    columns_to_use = ['classified_id','log_listing_price','brand_name','condition_name','subsubsubcategory_name']
    df = df[columns_to_use]

    # Return dataframe
    return df

def ridge_regression_pipeline(df):
    
    # Data transformation for ridge regression
    
    df = filter_rare_categories(df, 'subsubsubcategory_name', 'subsubcategory_name')
    df = condition_encoding(df)
    
    # New columns to use
    columns_to_use = ['classified_id', 'log_listing_price', 'condition_name','brand_name','subsubsubcategory_name']

    # Drop unused columns
    df = df[columns_to_use]

    # OHE columns
    df = pd.get_dummies(df, columns=['brand_name', 'subsubsubcategory_name'])
    
    # Extracting infrequent brands
    infrequent_brands = (df[[col for col in df.columns if 'brand' in col]].sum(axis=0).sort_values(ascending=True) < 30)
    infrequent_brands = infrequent_brands[infrequent_brands == True].index

    # Assigning 'other' to brands that are in infrequent_brands
    df['brand_name_other'] = df[infrequent_brands].sum(axis=1)
    df = df.drop(columns=infrequent_brands)

    return df