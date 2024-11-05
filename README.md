# Exploratory-Data-Analysis-on-Spotify-2023-Dataset

## Goal of this Project
In this Project you will perform an Exploratory Data Analysis (EDA) on an Dataset containing information of popular tracks on Most Streamed Spotify Songs 2023 (https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023). This task aims to analyze, visualize, and to intrepret data to both obtain and learn new valuable insights. 

## Objectives of Project and Guide Questions 
### 1. Overview of Dataset
- How many rows and columns does the dataset contain?
- What are the data types of each column? Are there any missing values?
### 2. Basic Descriptive Statistics
- What are the mean, median, and standard deviation of the streams column?
- What is the distribution of released_year and artist_count? Are there any noticeable trends or outliers?
### 3. Top Performers
- Which track has the highest number of streams? Display the top 5 most streamed tracks.
- Who are the top 5 most frequent artists based on the number of tracks in the dataset?
### 4. Temporal Trends
- Analyze the trends in the number of tracks released over time. Plot the number of tracks released per year.
- Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?
### 5. Genre and Music Characteristics
- Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%. Which attributes seem to influence streams the most?
- Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%?
### 6. Platform Popularity
- How do the numbers of tracks in spotify_playlists, spotify_charts, and apple_playlists compare? Which platform seems to favor the most popular tracks?
### 7. Advanced Analysis
- Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?
- Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.


## Codes Used, Ouput and Analysis
Note: Before Starting, download the file through the link then place the file in the same folder as your Jupyter notebook to be able to load the Data Frame

1. Importing all Libraries 
``` python
#Loading all different types of libraries
from math import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

2. Loading the Data Frame
``` python
#Loading the Dataset of Spotify 2023 and Output, latin1 is used to load the dataset properly 
spotify = pd.read_csv('spotify-2023.csv', encoding = 'latin1')
spotify
```
<img width="672" alt="image" src="https://github.com/user-attachments/assets/e9e85066-d44a-4e89-8f8f-c424581e5017">


3. Finding how many (rows,columns) is in the dataset
``` python
#Finding out how many rows and columns are there in the dataset (rows,columns)
spotify.shape
```
<img width="70" alt="image" src="https://github.com/user-attachments/assets/567f3266-c72e-4605-a380-c175f657b6a5">


4. Identifying the data type of each column 
```python
# Finding the Data type for each column 
spotify.info()
```
<img width="272" alt="image" src="https://github.com/user-attachments/assets/79e4bf36-870a-412d-9c10-f3e8a602049a">


5. Identifying if there are any Error(Null) Values in the dataset 
``` python
#Using "isnull" identify if there are any missing values(Null Values) in each column of the spotify Data set
pd.isnull(spotify).sum()
```
<img width="212" alt="image" src="https://github.com/user-attachments/assets/c308b8d7-6f12-47b1-b7d6-37003648c703">

6. Finding the mean, median, and standard deviation of the Streams column
```python
#Only accepting valid data and ignoring data that are not numerical value
spotify['streams'] = pd.to_numeric(spotify['streams'], errors = 'coerce')
#Getting the mean, median, and the Standard deviation 
a = spotify['streams'].mean()
b = spotify['streams'].median()
c = spotify['streams'].std()

# Printing the Final Output
print ("The mean, median, and standard deviation of the streams column")
print(f"Mean: {a}")
print(f"Median: {b}")
print(f"Standard deviation: {c}")
```
<img width="331" alt="image" src="https://github.com/user-attachments/assets/d332ab92-09d0-4114-8ba5-45e3f4902dbf">


7. Finding the dsitribution of released_year vs artist_count
```python
#Creating an Distribution plot for the released_year column with the color Red
sns.displot(spotify['released_year'], kde = False, color = '#FF0000')
#Title of Graph
plt.title('Distribution of Released year')
# Label for the y-axis 
plt.ylabel('Frequency')
# Label for the x-axis 
plt.xlabel('Released year')
# Outputing the Graph
plt.show()

#Creating an Distribution plot for the artist_count column with the color Blue
sns.displot(spotify['artist_count'], kde = False, color = 'blue')
#Title of Graph
plt.title('Distribution of Released year')
# Label for the y-axis 
plt.ylabel('Frequency')
# Label for the x-axis 
plt.xlabel('Artist Count')
# Outputing the Graph
plt.show()

#Creatting an scatter plot to find the relationship between the artist_count and released_year
#Making the Figure size
plt.figure(figsize = (6, 6))
#Getting the data in the data frame and assining them to their respective axis
sns.scatterplot(x = 'released_year', y = 'artist_count', data=spotify, color = 'green')
# The Title of the Graph
plt.title('Released Year vs Artist Count')
#The y-axis label
plt.ylabel('Artist Count')
#The x-axis label
plt.xlabel('Released Year')
#Outputting graph
plt.show()

#Note that the first 2 graphs shows the respected distribution of the given variables while the last graph shows the relationship between the two. 
```
 <img width="343" alt="image" src="https://github.com/user-attachments/assets/f85d94b9-1e8c-477b-beac-1dfcf5d429da">
 <img width="350" alt="image" src="https://github.com/user-attachments/assets/bfab0a6e-08ba-4fc0-b487-8f4c52952472">
 <img width="354" alt="image" src="https://github.com/user-attachments/assets/67195c94-91ec-4c30-934f-d206ca3d4ad2">

8. The Top 5 most Streamed Tracks
```python
# Sorting the column 'streams' from highest to lowest and choosing the first five rows
top_streamed_tracks = spotify.sort_values(by = 'streams', ascending = False).head(5)

# Pritning the title and Outputing the first 5 rows
print("\nTop 5 most streamed tracks:")
top_streamed_tracks
```
<img width="707" alt="image" src="https://github.com/user-attachments/assets/871e3bef-ca5b-4ee4-adb2-e9b336b2fbdc">


9. Top 5 most Frequent Artists
```python
# Counting the occurences of each artist name in the column then selecting the top 5
top_artist = spotify['artist(s)_name'].value_counts().head(5)
#Printing the title and the output 
print("\nTop 5 most frequent artists:")
top_artist
```
<img width="173" alt="image" src="https://github.com/user-attachments/assets/74fe74b3-3b44-4215-8c80-15d0b01ee220">


10. Number of tracks Released Yearly 
```python
#counting the number of tracks per year then turning into data frame
tracks_inyear = spotify['released_year'].value_counts().reset_index()
# Making an table and Rename the headings of the table 
tracks_inyear.columns = ['Released_year', 'Tracks']
#Printing the Heading the Data table
print ("Number of tracks per year")
print (tracks_inyear)

#Adjusting figure size
plt.figure(figsize = (14,6))
#Making an barplot
sns.barplot(x = 'Released_year', y = 'Tracks', data = tracks_inyear, color = 'green')
#x-axis label
plt.xlabel("Years")
#Adjusting x-axis labels
plt.xticks(rotation = 60)
#Printing Heading and outputing graph
plt.title("Number of tracks released per year")
plt.show()
```
<img width="691" alt="image" src="https://github.com/user-attachments/assets/3e386d94-1f78-49d1-8a2e-43536ddbe1ba">


11. Number of tracks released per month
```python
# Counting the number of tracks in each month
tracks_months = spotify['released_month'].value_counts().reset_index().sort_values(by = 'released_month')
# Dictionary assining the month to their respected month numeric value
month = { 1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December" }
#Mapping the tracks_months numeric value to each month name in the dictionary 
tracks_months['name_months'] = tracks_months['released_month'].map(month)
#Outputing the Table of Data 
print(tracks_months)

#Making a barplot with the color Red
sns.barplot(x = 'name_months', y = 'count', data = tracks_months, color = 'red')
#y-axis label 
plt.ylabel("Tracks")
#x-axis label 
plt.xlabel("Months")
#Rotating the x-axis label for better Visibility
plt.xticks(rotation = 60)
#Output Graph 
plt.show()
```
<img width="383" alt="image" src="https://github.com/user-attachments/assets/98397f0a-472b-479e-a641-b1634614490f">


12. The correlation between streams and musical attributes
```python
#Selecting the columns, then converting them to numerical values and assigning errors to 'coerce'
atrri = spotify[['streams', 'danceability_%', 'energy_%', 'valence_%', 'acousticness_%', 'instrumentalness_%', 'liveness_%', 'speechiness_%']].apply(pd.to_numeric, errors='coerce')
#Dropping all errors values 
clean = atrri.dropna()
#Calculating the correlation between each data 
cor = clean.corr()
# To find the correlation between streams and the musical atrributes then sorting them in descending order 
correl_streams = cor['streams'].sort_values(ascending = False)
#Printing the output
print (correl_streams)
```
<img width="207" alt="image" src="https://github.com/user-attachments/assets/c97e2ce3-33de-4ac0-9ff6-717f21fba82c">


13. Correlation between danceability_% vs energy_% and about valence_% vs acousticness_%
```python
# Finding the correlation between danceability_% vs energy_% and valence_% and acousticness_% 
#Converting all values to Numeric and Assigning all errors values to coerce so that it wont be counted
dance_energy = spotify[['danceability_%', 'energy_%']].apply(pd.to_numeric, errors='coerce').corr()
val_acous = spotify[['valence_%', 'acousticness_%']].apply(pd.to_numeric, errors='coerce').corr()
# Printing the results and Heading
print ('\nThe Correlation bewteen danceability_% and energy_% with respect to each sides')
print (dance_energy)
print ('\nThe Correlation bewteen valence_% and acousticness_% with respect to each sides')
print (val_acous)
```
<img width="434" alt="image" src="https://github.com/user-attachments/assets/3213a5d7-fcf9-4d32-9a64-7578caec09da">


14. The numbers of tracks in spotify_playlists, deezer_playlist, and apple_playlists
``` python
#Converting all values to numeric, to sum the number of tracks in each column category at the same time not counting any non numeric values 
count_spotify_playlists = spotify['in_spotify_playlists'].apply(pd.to_numeric, errors = 'coerce').sum()
count_deezer_playlist = spotify['in_deezer_playlists'].apply(pd.to_numeric, errors = 'coerce').sum()
count_apple_playlists = spotify['in_apple_playlists'].apply(pd.to_numeric, errors = 'coerce').sum()

#Outputing the Results 
print ("The number of tracks in spotify playlist is:", count_spotify_playlists)
print ("The number of tracks in deezer playlist is:", count_deezer_playlist)
print ("The number of tracks in apple playlist is:", count_apple_playlists)
```
<img width="287" alt="image" src="https://github.com/user-attachments/assets/4ed37f2c-004b-494b-a094-597952c9bf2a">


15. Stream data, identify pattern same key or mode (Major vs. Minor)
``` python
# Group by 'key' and 'mode', calculating the average streams then reset the index, and sorting the streams in descending order
stream = spotify.groupby(['key','mode'])['streams'].mean().reset_index().sort_values(by = 'streams', ascending = False)
# Printing Heading and Data table 
print("Average streams by key and mode")
print(stream)

#Define colors
color = ['Blue','Red']
#Making an Barplot with the new 'stream' data
sns.barplot(x = 'key', y = 'streams', hue = 'mode', data = stream, palette = color)
#Output Graph
plt.show()
```
<img width="403" alt="image" src="https://github.com/user-attachments/assets/7585a03f-370c-4d69-b191-fb8b677b3571">

16. Most frequently appearing artists in playlists or charts
``` python
#Coverting all columns inside the variable to numeric and naming all non numeric values to coerce "for the program to run"
conv = ['in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists', 'in_apple_charts', 'in_deezer_playlists', 'in_deezer_charts', 
                'in_shazam_charts']
spotify[conv] = spotify[conv].apply(pd.to_numeric, errors = 'coerce')

#Group data by artist name and sum of the numeric values across the chosen columns
artist_a = spotify.groupby('artist(s)_name')[conv].sum()
#Adding a 'Total' column and it contains the sums across the playlists 
artist_a['Total'] = artist_a.sum(axis = 1)
# Sorting the values in descending order and chosing the top 5
sort_artist = artist_a.sort_values(by = 'Total', ascending = False).reset_index().head()
#Assigning different colors in a variable 
color = ['black', 'blue', 'green', 'red', 'pink']
#Making an Barplot 
sns.barplot(x = 'artist(s)_name', y = 'Total', hue = 'Total', data = sort_artist, palette = color)
# x-axis label 
plt.xlabel('Artist Name')
#Output Graph
plt.show()
```
<img width="464" alt="image" src="https://github.com/user-attachments/assets/e720206b-dcce-4bcf-8ff3-28cd006d9587">


## Their Ouputs and Analysis

### Overview of Dataset
1. How many rows and columns does the dataset contain?
<img width="520" alt="image" src="https://github.com/user-attachments/assets/80f4452a-9e08-4f02-916a-7c1736fe86bc">

  Using the <ins>.shape</ins> code we are able to identify how many (rows,columns) there is in the dataset. 

2. What are the data types of each column? Are there any missing values?
<img width="237" alt="image" src="https://github.com/user-attachments/assets/72b6e927-00f2-4cb6-bbba-31989910139a">
<img width="176" alt="image" src="https://github.com/user-attachments/assets/7b2b9e93-e1e5-47f3-9e19-1aab60552e7f">

Using the <ins>.info()</ins> we were able to identify that there are 2 data types int64() and object() in the dataset. At the same time using <ins>.isnull()</ins> to identify if there are any null values and it was discovered that the key had 95 while in_shazam_charts had 50. 


### Basic Descriptive Statistics
3. What are the mean, median, and standard deviation of the streams column?
<img width="298" alt="image" src="https://github.com/user-attachments/assets/c6cfa5f1-8d74-429e-af69-ab757bdaa0f2">

Using the code we were able to get the mean, median, and standard deviation of the streams column. 

4. What is the distribution of released_year and artist_count? Are there any noticeable trends or outliers?
 <img width="343" alt="image" src="https://github.com/user-attachments/assets/f85d94b9-1e8c-477b-beac-1dfcf5d429da">
 <img width="350" alt="image" src="https://github.com/user-attachments/assets/bfab0a6e-08ba-4fc0-b487-8f4c52952472">
 <img width="354" alt="image" src="https://github.com/user-attachments/assets/67195c94-91ec-4c30-934f-d206ca3d4ad2">

After running the code it was noticable that as the years go by more artist starts to release more songs and do more collabs.


### Top Performers
5. Which track has the highest number of streams? Display the top 5 most streamed tracks.
<img width="547" alt="image" src="https://github.com/user-attachments/assets/b775f892-1e52-4cbd-aa09-566e644d323d">

This shows the Top 5 most streamed Songs

6. Who are the top 5 most frequent artists based on the number of tracks in the dataset?
<img width="200" alt="image" src="https://github.com/user-attachments/assets/79c1b598-2756-40ef-b1b3-5b384516508a">

This shows the Top 5 most Frequent artists based on the number of tracks


###  Temporal Trends
7. Analyze the trends in the number of tracks released over time. Plot the number of tracks released per year.
<img width="691" alt="image" src="https://github.com/user-attachments/assets/81ba4a76-a9c9-43c1-a9d9-b861846064d6">


It was observed that 2022 had the most tracks released in a year. As it can be seen in the graph above.
Note: If you wish to see the excat number of tracks in each year, visit my Jupyternotebook which can be seen in the same repository. 

8. Does the number of tracks released per month follow any noticeable patterns? Which month sees the most releases?
<img width="337" alt="image" src="https://github.com/user-attachments/assets/e0b3009a-7379-4ce4-9b61-7cc1957aa5f0">

It was observed that January had the most tracks in a month and there were no any noticeable patterns between the tracks releases per month. 


### Genre and Music Characteristics
9. Examine the correlation between streams and musical attributes like bpm, danceability_%, and energy_%. Which attributes seem to influence streams the most?
<img width="160" alt="image" src="https://github.com/user-attachments/assets/d71e6033-5e0f-4a89-9bea-c4ade91fdc93">

It can be seen that the only factors that can be seen to have an influence in the streams is danceability and speechiness even though its low, the rest of the factros don't really have an affect on the streams. 

10. Is there a correlation between danceability_% and energy_%? How about valence_% and acousticness_%?
<img width="391" alt="image" src="https://github.com/user-attachments/assets/98d19624-eeeb-4ffd-b3b6-106c03a93a2d">

There is a small correlation between danceability_% and energy_% though its a positive correlation, it looks to have an weak relationship. Meanwhile valence_% and acousticness_% have a weak negative correlation which means the higher valence, have a low acousticness however the relationship is weak. 


### Platform Popularity
11. How do the numbers of tracks in spotify_playlists, spotify_charts, and apple_playlists compare? Which platform seems to favor the most popular tracks?
<img width="317" alt="image" src="https://github.com/user-attachments/assets/9f070be4-29c9-4dc5-9213-5076de067d49">

It can be seen from the data above that the number of tracks in spotify playlist is the most popular platform with 4,955,719 tracks, while the number of tracks in deezer playlist comes in second then finally the number of tracks in the apple playlist being last. 


### Advanced Analysis
12. Based on the streams data, can you identify any patterns among tracks with the same key or mode (Major vs. Minor)?
<img width="428" alt="image" src="https://github.com/user-attachments/assets/12f53207-3d61-48e3-a938-0e68ef5b93f2">
<img width="232" alt="image" src="https://github.com/user-attachments/assets/d4968325-ab23-4723-b05d-da2115a0124a">

It can be seen from the data above that the most popular key is E Major with 7.605963e+08 streams while the least popular is G# minor with 3.219036e+08. There was no observed patterns that can be seen from the data. 

13. Do certain genres or artists consistently appear in more playlists or charts? Perform an analysis to compare the most frequently appearing artists in playlists or charts.
<img width="416" alt="image" src="https://github.com/user-attachments/assets/7f8ab7c8-69b8-4cea-9708-e320572e163f">

It can be observed that the Weekend is the most frequent artist being played in all platforms with an total of 150,273.0 plays, following Taylor swift and Ed Sheeran. This Code only shows the top 5 you can edit if you wish to. 

## Insights and Learnings 
  In the start of the coding it was difficult to load the dataset however after reaserching it was discovered that 'encoding = latin1' can be used to load the dataset to read the file correctly. In exploring the data frame it was obserevd that some values were considered errors so they were assignend to be 'coerce' so that they can be skipped and the data will resume running without any problems. It can be seen using the code at "Finding the mean, median, and standard deviation of the Streams column". Other than that other codes can be seen in the study lecture materials. Using a variaty of seaborn codes to present data in a graphical way to better visualise and compare each data or factor.

  In general, This porject was difficult and challenging for me however as i progressed in the work i learned and have gained knowledge when it comes to programming and have no doubt that in the future this knowledge will be used in my future profession. 


## Sources and Study References 
- ECE2112 Lecture Materials
- PYTHON PANDAS TUTORIAL #9 - HOW TO USE VALUE_COUNTS METHOD IN PANDAS. (https://www.youtube.com/watch?v=iYollp0FE_E )
- How to use groupby() to group categories in a pandas DataFrame (https://www.youtube.com/watch?v=ipoSjrN0oh0)
- 35 - Pandas - pandas.to_numeric() Method (https://www.youtube.com/watch?v=DrQzwmPr8Ts)
- Seaborn distplot | Seaborn distplot interpretation and how to make a distribution plot in seaborn (https://www.youtube.com/watch?v=fJoR3QsfXa0)
- Seaborn Tutorial : Seaborn Full Course (https://www.youtube.com/watch?v=6GUZXDef2U0)

## Author 
Kevin Matthew L. Santos
