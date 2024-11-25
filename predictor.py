import numpy as np
import pandas as pd
from data_loader import Loader

# Load pre-trained prediction matrix and data files
full_matrix = np.load('final_prediction_matrix.npy')
movies_df = pd.read_csv('movies.csv')  # Includes 'movieId', 'title', 'genres'
ratings_df = pd.read_csv('ratings.csv')  # Includes 'userId', 'movieId', 'rating', 'timestamp'

train_set = Loader(ratings_df)

def recommend_movies(user_id, top_n=3):
    """
    Generate movie recommendations for a given user.
    :param user_id: User ID to generate recommendations for.
    :param top_n: Number of top recommendations to return.
    :return: DataFrame with recommended movie titles and genres.
    """
    user_index = train_set.userid2idx.get(user_id, None)
    if user_index is None:
        return "User ID not found."

    # Get user ratings from the prediction matrix
    user_ratings = full_matrix[user_index]

    # Find top N movies for the user
    top_movie_indices = user_ratings.argsort()[-top_n:][::-1]
    top_movie_ids = [train_set.idx2movieid[idx] for idx in top_movie_indices]

    # Fetch movie titles and genres
    recommendations = movies_df[movies_df['movieId'].isin(top_movie_ids)][['title', 'genres']]
    return recommendations

    formatted_recommendations = recommendations.apply(
    lambda row: f"{str(row['movieId']).ljust(10)}  {row['title'].ljust(40)}  {row['genres']}",
    axis=1
    )
    return "\n".join(formatted_recommendations)