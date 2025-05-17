import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
movies = pd.read_csv('movies.csv')  # Contains movie titles and genres
ratings = pd.read_csv('ratings.csv')  # Contains user ratings

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Create a pivot table
pivot_table = data.pivot_table(index='userId', columns='title', values='rating')

# Fill NaN values with 0
pivot_table.fillna(0, inplace=True)

# Calculate cosine similarity
similarity = cosine_similarity(pivot_table)

# Create a DataFrame for similarity
sim_df = pd.DataFrame(similarity, index=pivot_table.index, columns=pivot_table.index)

# Function to recommend movies
def recommend_movies(user_id):
    similar_users = sim_df[user_id].sort_values(ascending=False).index[1:6]
    recommended_movies = pivot_table.loc[similar_users].mean().sort_values(ascending=False)
    return recommended_movies.head(5)

# Example usage
print(recommend_movies(user_id=1))
