import pandas as pd
imdb = pd.read_csv('imdb.csv')
imdb.head()



imdb.shape

Q1>  Find the average IMDb rating for movies released in  2010 and 2015

movies_2010_2015 = imdb[(imdb['Released_Year'] >= 2010) & (imdb['Released_Year'] <= 2015)]

# Calculate the sum of ratings and the number of movies,
sum_ratings_2010_2015 = movies_2010_2015['IMDB_Rating'].sum()
count_movies_2010_2015 = movies_2010_2015.shape[0]

# Calculate the average rating,  total sum of  Released_Year / no of rows
average_rating_2010_2015 = sum_ratings_2010_2015 / count_movies_2010_2015

print(f"The average IMDb rating for movies released between 2010 and 2015 is: {average_rating_2010_2015:.2f}")




Q2> Create a pie chart to visualize the distribution of average gross collection for movies belonging to the 'Sci-Fi', 'Action', and 'Drama' genres

# Calculate the average gross for each target genre across all movies containing that genre
target_genres = []
average_gross_target_genres = {}
for genre in target_genres:
    # Filter movies that contain the current target genre
    movies_with_genre = imdb[imdb['Genre'].str.contains(genre, na=False)]
    # Calculate the average gross for these movies
    average_gross_target_genres[genre] = movies_with_genre['Gross'].mean()

# Convert the dictionary to a pandas Series for easier plotting
average_gross_target_genres_series = pd.Series(average_gross_target_genres)

display(average_gross_target_genres_series)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
plt.pie(average_gross_target_genres_series, labels=average_gross_target_genres_series.index, autopct='%1.1f%%', startangle=140)
plt.title("Distribution of Average Gross for Sci-Fi, Action, and Drama Genres")
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

Q3> Compare the collection of Martin Scorsese Top 5 movies

scorsese_movies = imdb[imdb['Director'] == 'Martin Scorsese']
display(scorsese_movies.head())

top_5_scorsese_movies = scorsese_movies.sort_values(by='Gross', ascending=False).head(5)
display(top_5_scorsese_movies)

top_5_scorsese_collection = top_5_scorsese_movies[['title', 'Gross']]
display(top_5_scorsese_collection)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(top_5_scorsese_collection['title'], top_5_scorsese_collection['Gross'])
plt.xlabel('Movie Title')
plt.ylabel('Gross Collection')
plt.title('Top 5 Martin Scorsese Movies by Gross Collection')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

Q4> Create a histogram of the unique actors in the 'Star1' column for movies released between 2010 and 2015.

movies_2010_2015 = imdb[(imdb['Released_Year'] >= 2010) & (imdb['Released_Year'] <= 2015)]
display(movies_2010_2015.head())

star1_data = movies_2010_2015['Star1']
display(type(star1_data))
#d=star1_data.tolist()
#print(d)

actor_counts = star1_data.value_counts()
display(actor_counts.head())
print(type(actor_counts))
d=actor_counts.to_dict()
print(d)

plt.figure(figsize=(500, 6))
plt.bar(d.keys(),d.values())
#plt.hist(d)
plt.xlabel('Number of Movies as Star1')
plt.ylabel('Number of Actors')
plt.title('Distribution of Lead Actor Appearances (2010-2015)')

plt.show()