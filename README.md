# Phase 1 Project Description

## Project Overview
This project, will involve  data cleaning, imputation, analysis, and visualization to generate insights for the head of aviation in West Aviation Ltd, which is diversifying it's business by introducing an aviation business segment.
.

### Business context

West Aviation is expanding its potforlio to purchasing and operating airplanes for commercial and private enterprises.  They are in the process of undertaking a SWOT analysis in which pottential risks of aircraft is a key parameter. The objective of this analysis is determinning which aircraft are the lowest risk for the company to start this new business endeavor. 
The key output of the analysis is to translate the findings into actionable insights that the head of the new aviation division can use to help decide which aircraft to purchase.

### The Data Source

In the `data` folder is a [dataset](https://www.kaggle.com/datasets/khsamaha/aviation-accident-database-synopses) from the National Transportation Safety Board that includes aviation accident data from 1962 to 2023 about civil aviation accidents and selected incidents in the United States and international waters.

We used data/Aviation_Data.csv, to cleanse missing values, how to aggregate the data, and  to visualize it in an interactive dashboard.

## Deliverables

There are three deliverables for this project:

* A **non-technical presentation**
* A **Jupyter Notebook**
* A **GitHub repository**
* An **Interactive Dashboard**

# Data Understanding
## Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

#Loading the Aviation Data & Visualizing first 5 rows
df=pd.read_csv('data/Aviation_Data.csv')
df.head()

#Exploring data to identify the structure
df.info()
#From information summary above, many rows contain missing values. Below code visualizes the sum of all nulls per column.
df.isna().sum()
#Visualizing Null values as percentage for each column organized. This eases the view as well as cleansing approach 
null_percentage_sorted =  (df.isna().mean() * 100).round(1).sort_values(ascending=False)
print(null_percentage_sorted)

#Checking for whitespace in string columns
for col in df.select_dtypes(include='object').columns:
    has_whitespace = df[col].str.contains(r'^\s+|\s+$', na=False)
    if has_whitespace.any():
        print(f"Column '{col}' has  white spaces in {has_whitespace.sum()} rows.")

#From above code, we can see that the Location & Report status column has whitespace. To check unique values in the Location column.
print(df['Location'].unique())
print(df['Report.Status'].unique())

#Checking duplicate values in the data
duplicate_row = df[df.duplicated()]
df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_row.shape[0]}")
print(duplicate_row.head())

# Data Cleansing
From data exploration above, we have established that the data has missing values, whitespace in string columns(Location & Report.Status), Mixed pper/lower cases and duplicate rows.
Next step is to clean the data by filling and dropping missing values, changing datatypes for some columns, removing whitespaces as well as duplicates.
#1st we remove whitespace in string columns
df_cleansed = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
#Next we drop rows with any duplicate values
df_cleansed = df_cleansed.drop_duplicates()
#Next we drop columns with more than 60% null values 
threshold = len(df_cleansed) * 0.6
df_cleansed = df_cleansed.dropna(axis=1, thresh=threshold)
#Next we fill the remaining null values in the columns with less than 60% nulls
#For numeric columns, we fill with mean and for categorical columns, we fill with mode. 
for col in df_cleansed.columns:
    null_fraction = df_cleansed[col].isna().mean()
    if 0 < null_fraction < 0.6:
        if df_cleansed[col].dtype == "object":
            mode_val = df_cleansed[col].mode(dropna=True)
            if not mode_val.empty:
                df_cleansed[col] = df_cleansed[col].fillna(mode_val[0])
        else:
            df_cleansed[col] = df_cleansed[col].fillna(df_cleansed[col].mean())
 #Colum Make & Model have mixed cases. We can standardize them by converting to title case.
df_cleansed['Make'] = df_cleansed['Make'].str.strip().str.title()
df_cleansed.info()

#Checking for any remaining null values. Expectation is that there should be no null values left.
df_cleansed.isna().sum()  

#Checking for duplicate rows in the dataset. Expectation is that there should be no duplicates left.
df.duplicated()

# Checking for  white spaces in string columns. Expectation is that there should be no white spaces left.
for col in ['Location', 'Report.Status']:
 if col in df_cleansed.columns:
    has_whitespace = df_cleansed[col].str.contains(r'^\s+|\s+$', na=False)
    if has_whitespace.any():
        print(f"Column '{col}' has white spaces in {has_whitespace.sum()} rows.")
    else:
        print(f"Column '{col}' has no white spaces.")  

#Lastly, we convert some columns to appropriate datatypes
df_cleansed['Event.Date'] = pd.to_datetime(df_cleansed['Event.Date'], errors='coerce')
df_cleansed['Total.Fatal.Injuries'] = df_cleansed['Total.Fatal.Injuries'].astype('int')  
df_cleansed['Total.Minor.Injuries'] = df_cleansed['Total.Minor.Injuries'].astype('int')
df_cleansed['Total.Serious.Injuries'] = df_cleansed['Total.Serious.Injuries'].astype('int')
df_cleansed['Total.Fatal.Injuries'] = df_cleansed['Total.Fatal.Injuries'].astype('int')
df_cleansed['Number.of.Engines'] = df_cleansed['Number.of.Engines'].astype('int')
df_cleansed['Total.Uninjured'] = df_cleansed['Total.Uninjured'].astype('int')

#Checking the .info of the cleansed data. Expectation is that there should be no nulls, no duplicates, no white spaces and appropriate datatypes.
df_cleansed.info()  
df_cleansed.head(5)

# Exploratory Data Analysis
Now that we have cleansed data, the next phase is to analyse it based on business objectives. This includes aggregation and analysis of correlations amongst the variables to answer the question of what mix of factors should the head aviation consider to purchase the right planes to help achieve optimal risk as they seek to expand the business.
## 1. Which Make of Aircraft Model has the highest number of accidents? i.e Susceptible to accidents?
make_model_accidents = df_cleansed[['Make', 'Model']].value_counts().sort_values(ascending=False)
print("Aircraft Model with the highest number of accidents:")
print(make_model_accidents.head())

#Vizualizing the top 20 aircraft models with the highest number of accidents
import matplotlib.pyplot as plt
#visualize_make_model_accidents = make_model_accidents.head(20).plot(kind='bar', figsize=(12, 6), color='skyblue')
visualize_make_model_accidents = make_model_accidents.head(20).plot(
    kind='bar', 
    figsize=(10, 5), 
    color='skyblue'
)
plt.title('Top 20 Aircraft Models with the Highest Number of Accidents')
plt.ylabel('Number of Accidents')
plt.xlabel('Aircraft Make & Model')
plt.tight_layout()
plt.show()
##  Finding 1. 
Cessna Models have the highest number of accidents, followed by piper

## 2. Which Make of Aircraft is unsafe? i.e 
    # a) has the highest number of Injuries?
#To achieve this, we will create a new column 'Total.Injuries' as the sum of three columns: 'Total.Fatal.Injuries', 'Total.Minor.Injuries', and 'Total.Serious.Injuries'. This will help us analyze the total impact of accidents involving different aircraft makes.
df_cleansed['Total.Injuries'] = (
    df_cleansed['Total.Fatal.Injuries'] +
    df_cleansed['Total.Minor.Injuries'] +
    df_cleansed['Total.Serious.Injuries'] 
     
)

#Now we can group the data by 'Make' and sum the 'Total.Injuries' to find out which make has the highest number of accident injuries.
make_accidents = df_cleansed.groupby('Make')['Total.Injuries'].sum().sort_values(ascending=False)
print(make_accidents.head(20))

visualize_make_accidents = make_accidents.head(20).plot(
    kind='bar', 
    figsize=(10, 5), 
    color='coral'
)
plt.title('Top 20 Aircraft Makes with the Highest Number of Injuries in Accidents')
plt.ylabel('Number of Injuries')
plt.xlabel('Aircraft Make')
plt.tight_layout()
plt.show()

## Finding 2. 
Cessna has the highest number injuries.

## 3 which make and model of aircraft is the safest?
#a) Based people safety: To find the safest type of accident, we can analyze the 'Event.Type' column and the 'Total.Injuries' column. We will group the data by 'Event.Type' and sum the 'Total.Injuries' to find out which type of accident has the least number of injuries.
Most_Safe_aircraft=df_cleansed.groupby(['Make', 'Model'])['Total.Uninjured'].sum().sort_values(ascending=False)
print("Total uninjured per aircraft (Make & Model):")
print(Most_Safe_aircraft.head(10)) 
visualize_Most_Safe_aircraft = Most_Safe_aircraft.head(20).plot(
    kind='bar', 
    figsize=(10, 5), 
    color='green'
)
plt.title('Most safe Aircraft Models with lowest Number of Injuries')
plt.ylabel('Total Uninjured')
plt.xlabel('Aircraft Make & Model')
plt.tight_layout()
plt.show()
From the above, Boeing 737 is by far the safest in terms of uninjured
## b safety based on aircraft itself
    # based on Correlation between aircraft model and aircraft damage
#Count 'Destroyed' and 'Substantial' damages per Make & Model
damage_counts = df_cleansed.groupby(['Make', 'Model'])['Aircraft.damage'].value_counts().unstack(fill_value=0)

# Create a new column with the sum of 'Destroyed' and 'Substantial'
damage_counts['Destroyed_or_Substantial'] = damage_counts.get('Destroyed', 0) + damage_counts.get('Substantial', 0)

print("Destroyed + Substantial damage per aircraft model (top 10):")
print(damage_counts['Destroyed_or_Substantial'].sort_values(ascending=False).head(10))

## Which Boeing model is destroyed the most during accidents?
boeing_damage = damage_counts.loc['Boeing', 'Destroyed_or_Substantial']
print("Destroyed + Substantial damage per Boeing model:")
print(boeing_damage.sort_values(ascending=False))

 ## Finding 3: 
a) Boeing 737 is the safest plane by far in terms uninjured customers whenever an accident occurs, however, 737 is still the most destroyed boeing family whenever it is involved in the accident
b) Cessna is the most unsafe aircraft by design (in terms of magnitude of destruction) whenever an accident occurs

## Number of accidents per number of engines
accidents_per_model_engine = df_cleansed.groupby(['Make', 'Model', 'Number.of.Engines']).size().reset_index(name='Accident_Count')
print(accidents_per_model_engine.sort_values('Accident_Count', ascending=False).head(10))

# Plot visualization for number of accidents per aircraft and number of engines

top10 = accidents_per_model_engine.sort_values('Accident_Count', ascending=False).head(20)
plt.figure(figsize=(12,6))
plt.barh(
    top10.apply(lambda x: f"{x['Make']} {x['Model']} ({x['Number.of.Engines']} engines)", axis=1),
    top10['Accident_Count'],
    color='steelblue'
)
plt.xlabel('Number of Accidents')
plt.title('Top 10 Aircraft Models by Number of Accidents and Engines')
plt.gca().invert_yaxis()
plt.tight_layout()

accidents_per_model_engine_type = df_cleansed.groupby(['Make', 'Model', 'Engine.Type']).size().reset_index(name='Accident_Count')
print(accidents_per_model_engine_type.sort_values('Accident_Count', ascending=False).head(50))

accidents_per_model_engine_type = df_cleansed.groupby(['Make', 'Model', 'Engine.Type']).size().reset_index(name='Accident_Count')
top10 = accidents_per_model_engine_type.sort_values('Accident_Count', ascending=False).head(20)

plt.figure(figsize=(12,6))
plt.barh(
    top10.apply(lambda x: f"{x['Make']} {x['Model']} ({x['Engine.Type']})", axis=1),
    top10['Accident_Count'],
    color='slateblue'
)
plt.xlabel('Number of Accidents')
plt.title('Top 10 Aircraft Models by Number of Accidents and Engine Type')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

## Finding 4. 
a) Aircrafts with 1 Engine are prone to accidents that those with 2 engines.
b) Aircrafts with reciprocating and Turbo Shaft engine types are prone to accidents

#Correlation between accidents per country
country_accidents = df_cleansed['Country'].value_counts().sort_values(ascending=False)
print("Number of accidents per country:")
print(country_accidents.head(10))
#Visualize_country_accidents
plt.figure(figsize=(12, 6))
plt.scatter(country_accidents.head(20).index, country_accidents.head(20).values, color='indigo')
plt.title('Top 20 Countries with the Highest Number of Accidents')
plt.ylabel('Total Number of Accidents')
plt.xlabel('Country')
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()
## Finding 5:
Most accidents occurs in the United States.

# Reccomendations & Summary

Based on the correlation analysis infered above, this section gives recommendations to the company to consider as they review the risks involved in the venture.
## 1.  Based on finding 1 & 2 that Cessna Models have the highest number of:
 a) Accidents, 
 b) Highest number of injuries to passengers in cases of accidents
 b) Destroyed & Substantially destroyed aircrafts in cases of accidents
 
 Therefore, the company should minimize or avoid purchasing of Cessna aircrafts especially models 152, 172 & 172N

 ## 2. Based on finding 3:
Boeing is the safest aircraft as it has:
a) Highest uninjured passengers in case of accident
b) Relatively fewer aircrafts damaged during an accident

Therefore, the company should Maximize/optimize purchasing of Boeing aircrafts especially models 737.

## 3. Based on finding 4: Most accidents involve aircraft with 1 engine, Therefore the company should by aircrafts with more than 1 engine to reduce the risk of accidents.

## 3. Based on finding 5: Most accidents happen in the United, Therefore the company can review the possibilities of operating their aircrafts outside the united states.
