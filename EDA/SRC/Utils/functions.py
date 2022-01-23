

# creating variables for folders 

import sys, os
current_folder_location = os.getcwd()
SRC_FOLDER = os.path.dirname(current_folder_location)
EDA_FOLDER = os.path.dirname(SRC_FOLDER)
MAIN_FOLDER = os.path.dirname(EDA_FOLDER)


RAW_DATA = SRC_FOLDER + '\\DATA\\RAW\\'
PROCESSED_DATA = SRC_FOLDER + '\\DATA\\PROCESSED\\'



# Importing all libraries 
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import folium # pip install folium
from folium import plugins
import ipywidgets
import geocoder # pip install geocoder
import geopy # pip install geopy

from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
from numpy import cos, sin, arcsin, sqrt
from math import radians

from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
init_notebook_mode(connected=True)
import plotly.graph_objs as go
from branca.element import Figure

from vega_datasets import data as vds # pip install vega_datasets
import warnings
warnings.filterwarnings('ignore')


##################################################################################################################################################################################

#  Beginning of the cleaning DATA for properties Section

def load_cleaned_dataframe_before_2019(df1):

    #dropping columns that will not be used 
    df1.drop(['Serial Number','List Year', 'Assessed Value', 'Sales Ratio', 'Non Use Code', 'Assessor Remarks', 'OPM remarks', 'Location'], axis=1, inplace=True)

    #looking for null numbers
    df1.isna().sum()

    #filtering property only for residential 
    df1 = df1[df1['Property Type'] == 'Residential']

    #changing types of sales amount from float to int
    df1['Sale Amount'] = df1['Sale Amount'].astype(int)

    # DATE recorded to Datetime format 
    df1['Date Recorded'] = pd.to_datetime(df1['Date Recorded'])
    # replacing the values that do not make sense 
    df1['Sale Amount'] = df1['Sale Amount'].replace([10, 100, 487,500, 615], [130000, 40000, 97500, 310000, 181000])
    # getting rid of rows where is price is 0 and 1 
    df1[(df1['Sale Amount'] != 0) & (df1['Sale Amount'] != 1)]
    # deleting the column property type 
    df1 = df1.drop('Property Type', axis=1) 
    # renaming the row values 
    df1['Residential Type'] = df1['Residential Type'].replace(['Two Family','Three Family','Four Family'],['Multi-Family (2-4 Unit)', 'Multi-Family (2-4 Unit)', 'Multi-Family (2-4 Unit)'])
    
    # Renaming the column names 
    df1 = df1.rename(columns={'Date Recorded':'DATE SOLD', 'Town' : 'CITY', 'Address': 'ADDRESS', 'Sale Amount':'PRICE', 'Residential Type': 'PROPERTY TYPE'})
    # re arranging the column nammes to append with next one
    df1 = df1[['DATE SOLD','PROPERTY TYPE','ADDRESS','CITY','PRICE']]
    return df1

def load_cleaned_2nd_dataframe(df2):

    # deleting unnecessary columns
    df2.drop([ 'Unnamed: 0', 'SALE TYPE', 'BEDS', 'BATHS','STATE OR PROVINCE','SQUARE FEET', 'LOT SIZE','YEAR BUILT', 'DAYS ON MARKET', '$/SQUARE FEET', 'HOA/MONTH', 'STATUS', 'NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME', 'URL (SEE http://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)', 'SOURCE', 'MLS#', 'FAVORITE', 'INTERESTED', 'ZIP OR POSTAL CODE','LOCATION','LATITUDE', 'LONGITUDE' ], axis=1, inplace=True)

    #deleting the rows that consist of Null values
    df2 = df2.dropna()
    # Changing dtype from object to SOLD DATE 
    df2['SOLD DATE'] = pd.to_datetime(df2['SOLD DATE'])
    # Changing dtype from price from float to int 
    df2['PRICE'] = df2['PRICE'].astype(int)
    df2['ADDRESS'] = df2['ADDRESS'].str.upper()
    #renaming column types 
    df2['PROPERTY TYPE'] = df2['PROPERTY TYPE'].replace([ 'Single Family Residential' ,'Condo/Co-op'],['Single Family','Condo'])
    df2 = df2.rename(columns={'SOLD DATE': 'DATE SOLD'})
    # rearranging the column in order to append with the 
    df2 = df2[['DATE SOLD','PROPERTY TYPE','ADDRESS','CITY','PRICE']]

    return df2


def append_to_dataframes(df1, df2):
    df3 = df2.append(df1, ignore_index=True)

    # Starting Date is 1 Jan 2016
    df3 = df3[df3['DATE SOLD'] >= '2016-01-01']

    # below three lines will be to verify for any duplicates, if there are then clearing them out 
    df3.duplicated(subset=['ADDRESS','CITY','DATE SOLD']).sum()
    #print('Number of duplicates before deleting: ' + str(df3.duplicated(subset=['ADDRESS','CITY','DATE SOLD']).sum()))
    df3 = df3.drop_duplicates(subset=['ADDRESS','CITY','DATE SOLD'], keep='first')
    df3 = df3.dropna()
    df3['PROPERTY TYPE'] = df3['PROPERTY TYPE'].replace(['Multi-Family (2-4 Unit)', 'Multi-Family (5+ Unit)', 'Townhouse'],[ 'Multi-Family','Multi-Family', 'Multi-Family'])
    # Creating another table column for year sold in the column 
    df3['YEAR SOLD'] = df3['DATE SOLD'].dt.year

    return df3

# Splitting into three categories, Low, middle and high


def get_low_price_cat_houses(df3):
    low_price_cat_houses = df3[df3['PRICE'] <= 250000]
    return low_price_cat_houses

def get_middle_price_cat_houses(df3):
    middle_price_cat_houses = df3[(df3['PRICE'] >= 250000) & (df3['PRICE'] <= 550000)]
    return middle_price_cat_houses

def get_high_price_cat_houses(df3):
    luxury_price_cat_houses =  df3[df3['PRICE'] >= 550000]
    return luxury_price_cat_houses


# Prepearing DataFrames for types of properties Sold, Overall and During Pandemic 

def get_properties_sold_during_pandemic(df3):
    properties_sold_during_pandemic = df3[df3['DATE SOLD'] >= '2020-04-01']
    return properties_sold_during_pandemic


# End of cleaning DATA for properties Section 

#####################################################################################################################################################################################

# Beginnig of the Employment Section 

def load_general_CT_employment(DF_CT_general_employment_by_year):
    DF_CT_general_employment_by_year= DF_CT_general_employment_by_year.rename(columns={'CTUR': 'Unemployment Rate'})
    DF_CT_general_employment_by_year['Unemployment Rate'] = DF_CT_general_employment_by_year['Unemployment Rate'].round(1)
    DF_CT_general_employment_by_year['DATE'] = pd.to_datetime(DF_CT_general_employment_by_year['DATE'])
    DF_CT_general_employment_by_year['DATE'] = DF_CT_general_employment_by_year['DATE'].dt.year
    DF_CT_general_employment_by_year = DF_CT_general_employment_by_year[DF_CT_general_employment_by_year['DATE'] >= 2016]

    return DF_CT_general_employment_by_year


def load_general_CT_employment_by_month(DF_CT_general_employment_by_month):
    DF_CT_general_employment_by_month.rename(columns={'CTUR': 'Unemployment Rate'}, inplace=True)
    #Changing from object date column to datetime object 
    DF_CT_general_employment_by_month['DATE'] = pd.to_datetime(DF_CT_general_employment_by_month['DATE'])
    DF_CT_general_employment_by_month = DF_CT_general_employment_by_month[DF_CT_general_employment_by_month['DATE'] >= '2020-01-01']
    
    return DF_CT_general_employment_by_month


def load_general_CT_employment_by_year_upgraded(DF_CT_general_employment_by_year, DF_CT_general_employment_by_month):
    #creating a new variable with date 1 Jan 2020
    overall_ct_report_by_month_Jan_2021 = DF_CT_general_employment_by_month[DF_CT_general_employment_by_month['DATE'] == '2021-01-01'] 
    overall_ct_report_by_month_Jan_2021['DATE'] = overall_ct_report_by_month_Jan_2021['DATE'].dt.year

    #appending the original df employment by year with the overall_ct_report_by_month_Jan_2021
    overall_ct_employment_by_year = DF_CT_general_employment_by_year.append(overall_ct_report_by_month_Jan_2021, ignore_index=True)

    #Changing the Date column with year  
    overall_ct_employment_by_year = overall_ct_employment_by_year.rename(columns={'DATE': 'YEAR'})

    return overall_ct_employment_by_year

# End of employment section 

#####################################################################################################################################################################################

# Beginning of the Income section 

def load_clean_CT_income_by_year(df_income_CT_by_year):
    # ranmeing the column name 
    df_income_CT_by_year.rename(columns={'DATE':'YEAR', 'MEHOINUSCTA646N':'CT MEDIAN HOUSEHOLD INCOME'}, inplace=True)
    # converting to datetime 
    df_income_CT_by_year['YEAR'] = pd.to_datetime(df_income_CT_by_year['YEAR'])
    # extracting only a year 
    df_income_CT_by_year['YEAR'] = df_income_CT_by_year['YEAR'].dt.year
    # 2016 or later
    df_income_CT_by_year = df_income_CT_by_year[df_income_CT_by_year['YEAR'] >= 2016]
    return df_income_CT_by_year


def load_clean_CT_income_per_state_by_year(df_income_CT_per_state_by_year):
    # deleting unnecessary columns 
    df_income_CT_per_state_by_year.drop(['Unnamed: 0', '2','4','6','8','10','12','14'], axis=1, inplace=True)
    # renaming columns 
    df_income_CT_per_state_by_year.rename(columns={'0':'YEAR', '1': 'Fairfield County', '3': 'Hartford County', '5': 'Litchfield County', '7': 'Middlessex County', '9': 'New_haven County', '11': 'New London County', '13': 'Tolland County', '15': 'Windham County'}, inplace=True)

    df_income_CT_per_state_by_year['YEAR'] = pd.to_datetime(df_income_CT_per_state_by_year['YEAR'])

    # extracting only year from the column 
    df_income_CT_per_state_by_year['YEAR'] = df_income_CT_per_state_by_year['YEAR'].dt.year
    # Starting from year 2016 
    df_income_CT_per_state_by_year = df_income_CT_per_state_by_year[df_income_CT_per_state_by_year['YEAR'] >= 2016]
    return df_income_CT_per_state_by_year

def load_CT_median_house_hold_income_vs_fairfield_county(df_income_CT_by_year, df_income_CT_per_state_by_year):
    CT_general_income_last_5_years_to_compare_with_fairfield = df_income_CT_by_year[:-1]
    income_by_county_last_5_years_only_fairfield = df_income_CT_per_state_by_year.iloc[:, 0:2]

    CT_income_VS_fairfield_county = pd.merge(left=CT_general_income_last_5_years_to_compare_with_fairfield, right=income_by_county_last_5_years_only_fairfield, how="inner", on="YEAR")

    return CT_income_VS_fairfield_county

# End of Income section 

###################################################################################################################################################################################

# Beginning of the Data Analysis in General 

def get_number_of_sales_by_year(df3):
    n_sales = df3[['YEAR SOLD','DATE SOLD']].groupby(['YEAR SOLD']).count()
    n_sales_by_year = n_sales.rename(columns={'DATE SOLD': 'Number of Sales'})
    #n_sales_by_year = n_sales_by_year.reset_index()

    return n_sales_by_year

    
def get_n_of_sales_by_city(df3):
    n_sales_by_CITY_top10 = df3[['CITY','DATE SOLD']].groupby(['CITY']).count().sort_values('DATE SOLD', ascending=False).head(10)
    n_sales_by_CITY_top10 = n_sales_by_CITY_top10.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_CITY_top10 = n_sales_by_CITY_top10.reset_index()
    n_sales_by_CITY_top10

    return n_sales_by_CITY_top10


# This function below will be used to get top 20 Cities, it will be used to create a dataframe for the wordClou
def get_n_of_sales_by_city_top_20(df3):
    n_sales_by_CITY_top20 = df3[['CITY','DATE SOLD']].groupby(['CITY']).count().sort_values('DATE SOLD', ascending=False).head(20)
    n_sales_by_CITY_top20 = n_sales_by_CITY_top20.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_CITY_top20 = n_sales_by_CITY_top20.reset_index()
    n_sales_by_CITY_top20 

    n_sales_by_CITY_top20['CITY'] = n_sales_by_CITY_top20['CITY'].replace(['West Hartford','East Hartford','New Britain', 'New Haven'],['West_Hartford', 'East_Hartford', 'New_Britain', 'New_Haven'])
   
    return n_sales_by_CITY_top20


# Getting Data analysis based on Price Categories 

# Getting number of Sales by Year 

def get_number_of_sales_by_year_low_price(low_price_cat_houses):
    n_sales_low_price = low_price_cat_houses[['YEAR SOLD','DATE SOLD']].groupby(['YEAR SOLD']).count()
    n_sales_by_year_low_price = n_sales_low_price.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_year_low_price = n_sales_by_year_low_price.reset_index()

    return n_sales_by_year_low_price

def get_number_of_sales_by_year_middle_price(middle_price_cat_houses):
    n_sales_middle_price = middle_price_cat_houses[['YEAR SOLD','DATE SOLD']].groupby(['YEAR SOLD']).count()
    n_sales_by_year_middle_price = n_sales_middle_price.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_year_middle_price = n_sales_by_year_middle_price.reset_index()

    return n_sales_by_year_middle_price


def get_number_of_sales_by_year_high_price(luxury_price_cat_houses):
    n_sales_high_price = luxury_price_cat_houses[['YEAR SOLD','DATE SOLD']].groupby(['YEAR SOLD']).count()
    n_sales_by_year_high_price = n_sales_high_price.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_year_high_price = n_sales_by_year_high_price.reset_index()

    return n_sales_by_year_high_price


# Adding these Dataframes together to make a comparison


def add_n_sales_by_categories_for_comparison(n_sales_by_year_low_price, n_sales_by_year_middle_price, n_sales_by_year_high_price):
    
    n_sales_by_year_low_price_copied = n_sales_by_year_low_price.copy()
    n_sales_by_year_low_price_copied.rename(columns={'Number of Sales': 'LOW PRICE'}, inplace=True)
    
    n_sales_by_year_middle_price_copied = n_sales_by_year_middle_price.copy()
    n_sales_by_year_middle_price_copied.rename(columns={'Number of Sales': 'MIDDLE PRICE'}, inplace=True)
    
    n_sales_by_year_high_price_copied = n_sales_by_year_high_price.copy()
    n_sales_by_year_high_price_copied.rename(columns={'Number of Sales': 'LUXURY PRICE'}, inplace=True)



    top10_middle_luxury = [n_sales_by_year_middle_price_copied, n_sales_by_year_high_price_copied]

    for top10_mid_lux in top10_middle_luxury:
        for column in top10_mid_lux.columns:
            if "YEAR SOLD" in column:
                top10_mid_lux.drop(column, axis=1, inplace=True)
        

    number_of_sales_with_index_by_years_split_by_price_categories = pd.concat([n_sales_by_year_low_price_copied, n_sales_by_year_middle_price_copied, n_sales_by_year_high_price_copied], axis=1)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index = number_of_sales_with_index_by_years_split_by_price_categories.set_index('YEAR SOLD')

    return number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index

def graph_add_n_sales_by_categories_for_comparison(number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index):
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = plt.figure(figsize=(10,6), dpi=100)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = plt.xlabel('Year', fontsize=18)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = plt.ylabel('Number of sales', fontsize=18)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = plt.xticks(fontsize=14)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = plt.yticks(fontsize=14)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = plt.ylim(0, 27000)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = plt.title('LOW PRICE VS MIDDLE PRICE VS LUXURY PRICE', fontsize=16)
    number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph = sns.lineplot(data=number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index, markers=True)
    return number_of_sales_with_index_by_years_split_by_price_categories_with_YEAR_SOLD_index_graph

#################################################################################################################################################################

# Getting number of sales by CIty for three different categories, Below are just DataFrames

def get_n_of_sales_by_city_low_price(low_price_cat_houses):
    n_sales_by_CITY_top10_low_price = low_price_cat_houses[['CITY','DATE SOLD']].groupby(['CITY']).count().sort_values('DATE SOLD', ascending=False).head(10)
    n_sales_by_CITY_top10_low_price = n_sales_by_CITY_top10_low_price.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_CITY_top10_low_price = n_sales_by_CITY_top10_low_price.reset_index()
    n_sales_by_CITY_top10_low_price

    return n_sales_by_CITY_top10_low_price

def get_n_of_sales_by_city_middle_price(middle_price_cat_houses):
    n_sales_by_CITY_top10_middle_price = middle_price_cat_houses[['CITY','DATE SOLD']].groupby(['CITY']).count().sort_values('DATE SOLD', ascending=False).head(10)
    n_sales_by_CITY_top10_middle_price = n_sales_by_CITY_top10_middle_price.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_CITY_top10_middle_price = n_sales_by_CITY_top10_middle_price.reset_index()
    n_sales_by_CITY_top10_middle_price

    return n_sales_by_CITY_top10_middle_price

def get_n_of_sales_by_city_middle_price(luxury_price_cat_houses):
    n_sales_by_CITY_top10_luxury_price = luxury_price_cat_houses[['CITY','DATE SOLD']].groupby(['CITY']).count().sort_values('DATE SOLD', ascending=False).head(10)
    n_sales_by_CITY_top10_luxury_price = n_sales_by_CITY_top10_luxury_price.rename(columns={'DATE SOLD': 'Number of Sales'})
    n_sales_by_CITY_top10_luxury_price = n_sales_by_CITY_top10_luxury_price.reset_index()
    n_sales_by_CITY_top10_luxury_price

    return n_sales_by_CITY_top10_luxury_price


#####################################################################################################################################################################################

# Beginning of the plotting graphs for Analysis overall 

def graph_number_of_sales_by_year(df3, n_sales_by_year):
    Years = [year for year, df_ in df3.groupby('YEAR SOLD')]

    number_of_sales_by_year_graph = plt.figure(figsize=(8,6), dpi = 100)

    number_of_sales_by_year_graph = plt.bar(Years, n_sales_by_year['Number of Sales'])
    number_of_sales_by_year_graph = plt.xticks(Years, rotation=45, fontsize=14)
    number_of_sales_by_year_graph = plt.yticks(fontsize=14)
    number_of_sales_by_year_graph = plt.xlabel('Year', fontsize=16)
    number_of_sales_by_year_graph = plt.ylabel('Number of Sales', fontsize=16)
    number_of_sales_by_year_graph = plt.ylim(0,47000)
    number_of_sales_by_year_graph = plt.title('Number of sales by Year', fontsize=18)

    return number_of_sales_by_year_graph

def graph_the_unemployment_rate_by_year(overall_ct_employment_by_year):
    unemployment_rate = overall_ct_employment_by_year['Unemployment Rate']
    year = overall_ct_employment_by_year['YEAR']

    # Create bars
    unemployment_rate_by_year_graph = plt.figure(figsize=(8,6), dpi=100)
    unemployment_rate_by_year_graph = plt.bar(year, unemployment_rate)
    unemployment_rate_by_year_graph = plt.xlabel('Year', fontsize=16)
    unemployment_rate_by_year_graph = plt.ylabel('Unemployment rate', fontsize=16)
    # Create names on the x-axis
    unemployment_rate_by_year_graph = plt.xticks(year, rotation=70, fontsize=14)
    unemployment_rate_by_year_graph = plt.yticks(fontsize=14)
    unemployment_rate_by_year_graph = plt.title('Unemployment Rate by Year', fontsize=18)

    return unemployment_rate_by_year_graph

def graph_the_unemployment_rate_last_two_years(DF_CT_general_employment_by_month):

    unemployment_last_2_years = plt.figure(figsize=(8,6), dpi=100)
    unemployment_last_2_years = sns.lineplot(x = "DATE", y = "Unemployment Rate", data = DF_CT_general_employment_by_month)
    unemployment_last_2_years = plt.xticks(rotation=60, fontsize=14)
    unemployment_last_2_years = plt.yticks(fontsize=14)
    unemployment_last_2_years = plt.xlabel('Year-Month', fontsize=16)
    unemployment_last_2_years = plt.ylabel('Unemployment Rate', fontsize=16)
    unemployment_last_2_years = plt.title('Unemployment Rate (last 2 years) ', fontsize=18)

    return unemployment_last_2_years


def graph_n_of_sales_by_city(n_sales_by_CITY_top10):
    # plot the result
    n_sales_by_CITY = plt.figure(figsize=(10,6), dpi=100)
    n_sales_by_CITY = plt.xticks(rotation=22, fontsize=14, ha='right')
    n_sales_by_CITY = plt.yticks(fontsize=14 )
    n_sales_by_CITY = sns.barplot(x = n_sales_by_CITY_top10["CITY"], y = n_sales_by_CITY_top10['Number of Sales'])
    n_sales_by_CITY = plt.xlabel('City', fontsize=16)
    n_sales_by_CITY = plt.ylabel('Number of Sales', fontsize=16)
    n_sales_by_CITY = plt.title('Number of sales by Cities (Top 10)', fontsize=18 )

    return n_sales_by_CITY


#Plotting graphs for analysis based on price category

def graph_n_sales_by_year_low_price(n_sales_by_year_low_price):
    # plot the result
    n_sales_by_year_low_price_graph = plt.figure(figsize=(8,6), dpi=100)
    n_sales_by_year_low_price_graph = plt.xticks(rotation=22, fontsize=14, ha='right')
    n_sales_by_year_low_price_graph = plt.yticks(fontsize=14 )
    n_sales_by_year_low_price_graph = sns.barplot(x = n_sales_by_year_low_price["YEAR SOLD"], y = n_sales_by_year_low_price['Number of Sales'])
    n_sales_by_year_low_price_graph = plt.xlabel('Year', fontsize=20)
    n_sales_by_year_low_price_graph = plt.ylabel('Number of Sales', fontsize=20)
    n_sales_by_year_low_price_graph = plt.title('Number of sales by Year (Low)')

    return n_sales_by_year_low_price_graph 


def graph_n_sales_by_year_middle_price(n_sales_by_year_middle_price):
    # plot the result
    n_sales_by_year_middle_price_graph = plt.figure(figsize=(8,6), dpi=100)
    n_sales_by_year_middle_price_graph = plt.xticks(rotation=22, fontsize=14, ha='right')
    n_sales_by_year_middle_price_graph = plt.yticks(fontsize=14 )
    n_sales_by_year_middle_price_graph = sns.barplot(x = n_sales_by_year_middle_price["YEAR SOLD"], y = n_sales_by_year_middle_price['Number of Sales'])
    n_sales_by_year_middle_price_graph = plt.xlabel('Year', fontsize=20)
    n_sales_by_year_middle_price_graph = plt.ylabel('Number of Sales', fontsize=20)
    n_sales_by_year_middle_price_graph = plt.title('Number of sales by Year (Middle)')

    return n_sales_by_year_middle_price_graph

def graph_n_sales_by_year_high_price(n_sales_by_year_high_price):
    # plot the result
    
    n_sales_by_year_high_price_graph = plt.figure(figsize=(8,6), dpi=100)
    n_sales_by_year_high_price_graph = plt.xticks(rotation=22, fontsize=14, ha='right')
    n_sales_by_year_high_price_graph = plt.yticks(fontsize=14 )
    n_sales_by_year_high_price_graph = sns.barplot(x = n_sales_by_year_high_price["YEAR SOLD"], y = n_sales_by_year_high_price['Number of Sales'])
    n_sales_by_year_high_price_graph = plt.xlabel('Year', fontsize=20)
    n_sales_by_year_high_price_graph = plt.ylabel('Number of Sales', fontsize=20)
    n_sales_by_year_high_price_graph = plt.title('Number of sales by Year (Luxury)')

    return n_sales_by_year_high_price_graph


#Below are the graphs for N of Sales by City per different categories

def graph_n_of_sales_by_city_low_price(n_sales_by_CITY_top10_low_price):
    # plot the result
    n_sales_by_CITY_low_price_graph = plt.figure(figsize=(10,6), dpi=100)
    n_sales_by_CITY_low_price_graph = plt.xticks(rotation=22, fontsize=14, ha='right')
    n_sales_by_CITY_low_price_graph = plt.yticks(fontsize=14 )
    n_sales_by_CITY_low_price_graph = sns.barplot(x = n_sales_by_CITY_top10_low_price["CITY"], y = n_sales_by_CITY_top10_low_price['Number of Sales'])
    n_sales_by_CITY_low_price_graph = plt.xlabel('City', fontsize=16)
    n_sales_by_CITY_low_price_graph = plt.ylabel('Number of Sales', fontsize=16)
    n_sales_by_CITY_low_price_graph = plt.title('Number of sales by Cities (Low)', fontsize=18 )

    return n_sales_by_CITY_low_price_graph

def graph_n_of_sales_by_city_middle_price(n_sales_by_CITY_top10_middle_price):
    # plot the result
    n_sales_by_CITY_middle_price_graph = plt.figure(figsize=(10,6), dpi=100)
    n_sales_by_CITY_middle_price_graph = plt.xticks(rotation=22, fontsize=14, ha='right')
    n_sales_by_CITY_middle_price_graph = plt.yticks(fontsize=14 )
    n_sales_by_CITY_middle_price_graph = sns.barplot(x = n_sales_by_CITY_top10_middle_price["CITY"], y = n_sales_by_CITY_top10_middle_price['Number of Sales'])
    n_sales_by_CITY_middle_price_graph = plt.xlabel('City', fontsize=16)
    n_sales_by_CITY_middle_price_graph = plt.ylabel('Number of Sales', fontsize=16)
    n_sales_by_CITY_middle_price_graph = plt.title('Number of sales by Cities (Middle)', fontsize=18 )

    return n_sales_by_CITY_middle_price_graph

def graph_n_of_sales_by_city_lux_price(n_sales_by_CITY_top10_luxury_price):
    # plot the result
    n_sales_by_CITY_lux_price_graph = plt.figure(figsize=(10,6), dpi=100)
    n_sales_by_CITY_lux_price_graph = plt.xticks(rotation=22, fontsize=14, ha='right')
    n_sales_by_CITY_lux_price_graph = plt.yticks(fontsize=14 )
    n_sales_by_CITY_lux_price_graph = sns.barplot(x = n_sales_by_CITY_top10_luxury_price["CITY"], y = n_sales_by_CITY_top10_luxury_price['Number of Sales'])
    n_sales_by_CITY_lux_price_graph = plt.xlabel('City', fontsize=16)
    n_sales_by_CITY_lux_price_graph = plt.ylabel('Number of Sales', fontsize=16)
    n_sales_by_CITY_lux_price_graph = plt.title('Number of sales by Cities (Luxury)', fontsize=18 )

    return n_sales_by_CITY_lux_price_graph



# Below will be dataframes for Types of property, 1st Overall, 2nd During Pandemic 




def count_property_types_sold(df3):
    n_of_sales_based_on_properties = df3[['PROPERTY TYPE', 'ADDRESS']].groupby('PROPERTY TYPE').count()
    n_of_sales_based_on_properties = n_of_sales_based_on_properties.rename(columns={'ADDRESS': 'Number_of_Sales'})
    n_of_sales_based_on_properties = n_of_sales_based_on_properties.reset_index()
    
    return n_of_sales_based_on_properties


def count_property_types_sold_during_pandemic(properties_sold_during_pandemic):
    n_of_sales_based_on_properties_during_pandemic = properties_sold_during_pandemic[['PROPERTY TYPE', 'ADDRESS']].groupby('PROPERTY TYPE').count()
    n_of_sales_based_on_properties_during_pandemic = n_of_sales_based_on_properties_during_pandemic.rename(columns={'ADDRESS': 'Number_of_Sales'})
    n_of_sales_based_on_properties_during_pandemic = n_of_sales_based_on_properties_during_pandemic.reset_index()
    
    return n_of_sales_based_on_properties_during_pandemic



# getting Lottitude and Lattitude for top 10 cities based on N of Sales 

def get_long_lat_for_top10_cities_sold(n_sales_by_CITY):
    n_sales_by_CITY_with_lat_long = n_sales_by_CITY.copy()
    n_sales_by_CITY_with_lat_long['STATE'] = n_sales_by_CITY['CITY'].apply(lambda x: x + ' ' + 'Connecticut') 
    
    # declare an empty list to store
    # latitude and longitude of values 
    # of city column
    longitude = []
    latitude = []
    
    # function to find the coordinate
    # of a given city 
    def findGeocode(city):
        
        # try and catch is used to overcome
        # the exception thrown by geolocator
        # using geocodertimedout  
        try:
            
            # Specify the user_agent as your
            # app name it should not be none
            geolocator = Nominatim(user_agent="your_app_name")
            
            return geolocator.geocode(city)
        
        except GeocoderTimedOut:
            
            return findGeocode(city)    
    
    # each value from city column
    # will be fetched and sent to
    # function find_geocode   
    for i in (n_sales_by_CITY_with_lat_long["STATE"]):
        
        if findGeocode(i) != None:
            
            loc = findGeocode(i)
            
            # coordinates returned from 
            # function is stored into
            # two separate list
            latitude.append(loc.latitude)
            longitude.append(loc.longitude)
        
        # if coordinate for a city not
        # found, insert "NaN" indicating 
        # missing value 
        else:
            latitude.append(np.nan)
            longitude.append(np.nan)
    #Showing the output produced as dataframe.

    # now add this column to dataframe
    n_sales_by_CITY_with_lat_long["Longitude"] = longitude
    n_sales_by_CITY_with_lat_long["Latitude"] = latitude
    

    n_sales_by_CITY_with_lat_long = n_sales_by_CITY_with_lat_long.drop('STATE', axis=1)

    return n_sales_by_CITY_with_lat_long


# calculating the KM based LONG and LAT 

def haversine(row):
    lon1 = -73.935242
    lat1 = 40.730610
    lon2 = row['Longitude']
    lat2 = row['Latitude']
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * arcsin(sqrt(a)) 
    km = 6367 * c
    
    return km

def get_distance_from_NY(n_sales_by_CITY_with_lat_long):
    n_sales_by_CITY_with_lat_long['Distance_from_New_York (km)'] = n_sales_by_CITY_with_lat_long.apply(lambda row: haversine(row), axis=1)
    n_sales_by_CITY_with_lat_long_dist_from_NY = n_sales_by_CITY_with_lat_long[n_sales_by_CITY_with_lat_long['Distance_from_New_York (km)'] < 100] 
    return n_sales_by_CITY_with_lat_long_dist_from_NY

def show_map_of_cities_for_sales_less_than_100km_ny(n_sales_by_CITY_with_lat_long_dist_from_NY):
    
    map_top10_cities_less_than_100km_from_NY = folium.Map(location=[41.1095336616298, -73.60959505779903], zoom_start=9.6)

    # plot cities locations 
    for (index, row) in n_sales_by_CITY_with_lat_long_dist_from_NY.iterrows():
        folium.Marker(location = [row.loc['Latitude'], row.loc['Longitude']],
                    popup = 'Number of Sales: ' + str(row.loc['Number of Sales']) + ' ' + 'Distance from NYC: ' + str(row['Distance_from_New_York (km)']),
                    tooltip = 'click').add_to(map_top10_cities_less_than_100km_from_NY)
    
    fig_map_CT = Figure(width=1000, height=920)
    fig_map_CT.add_child(map_top10_cities_less_than_100km_from_NY)
    return fig_map_CT



# calculating types of properties sold 

def pie_chart_of_sales_by_cat(df, col1, col2):
    group_by_category_pie_chart = plt.figure(figsize=(10,6), dpi=100)
    group_by_category_pie_chart.patch.set_facecolor('white')
    group_by_category_pie_chart = plt.pie(df[col1], labels = df[col2],autopct='%1.1f%%')
    group_by_category_pie_chart = plt.title('Type of Properties Sold', fontsize=18)

    return group_by_category_pie_chart

    
def show_income_of_all_counties(df_income_CT_per_state_by_year):

    df_income_CT_per_state_by_year_for_graph = df_income_CT_per_state_by_year.copy()
    df_income_CT_per_state_by_year_for_graph['YEAR'] = df_income_CT_per_state_by_year_for_graph['YEAR'].astype(str)
    df_all_counties_income_together_index = df_income_CT_per_state_by_year_for_graph.set_index('YEAR')
    df_income_CT_per_state_by_year_visualization = plt.figure(figsize=(10,6), dpi=100)
    df_income_CT_per_state_by_year_visualization = plt.xlabel('Year', fontsize=18)
    df_income_CT_per_state_by_year_visualization = plt.ylabel('Median household Income', fontsize=18)
    df_income_CT_per_state_by_year_visualization = plt.xticks(fontsize=14)
    df_income_CT_per_state_by_year_visualization = plt.yticks(fontsize=14)
    df_income_CT_per_state_by_year_visualization = plt.ylim(55000, 100000)
    df_income_CT_per_state_by_year_visualization = plt.title('Median Household Income per County', fontsize=18)
    df_income_CT_per_state_by_year_visualization = sns.lineplot(data=df_all_counties_income_together_index, markers=True)
    df_income_CT_per_state_by_year_visualization = plt.legend(bbox_to_anchor=(1.01, 0.72), borderaxespad=0)
    

    return df_income_CT_per_state_by_year_visualization


def show_fairfield_income_vs_CT_median(CT_income_VS_fairfield_county):
    CT_income_VS_fairfield_county_for_graph = CT_income_VS_fairfield_county.copy()
    CT_income_VS_fairfield_county_for_graph['YEAR'] = CT_income_VS_fairfield_county_for_graph['YEAR'].astype(str)
    CT_income_VS_fairfield_county_for_graph_with_index = CT_income_VS_fairfield_county_for_graph.set_index('YEAR')
    CT_income_VS_fairfield_county_visualization = plt.figure(figsize=(10,6), dpi=100)
    CT_income_VS_fairfield_county_visualization = plt.xlabel('Year', fontsize=18)
    CT_income_VS_fairfield_county_visualization = plt.ylabel('Median household Income', fontsize=18)
    CT_income_VS_fairfield_county_visualization = plt.xticks(fontsize=14)
    CT_income_VS_fairfield_county_visualization = plt.yticks(fontsize=14)
    CT_income_VS_fairfield_county_visualization = plt.ylim(55000, 100000)
    CT_income_VS_fairfield_county_visualization = plt.title('CT vs Fairfield County', fontsize=18)
    CT_income_VS_fairfield_county_visualization = sns.lineplot(data=CT_income_VS_fairfield_county_for_graph_with_index, markers=True)
    CT_income_VS_fairfield_county_visualization = plt.legend(loc='lower right')

    return CT_income_VS_fairfield_county_visualization