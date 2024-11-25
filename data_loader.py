import torch
import pandas as pd
from torch.utils.data import Dataset

class Loader(Dataset):
    def __init__(self, ratings_df):
        self.ratings = ratings_df.copy()

        # Extract unique users and movies
        users = ratings_df.userId.unique()
        movies = ratings_df.movieId.unique()

        # Create mappings for user and movie IDs
        self.userid2idx = {user_id: idx for idx, user_id in enumerate(users)}
        self.movieid2idx = {movie_id: idx for idx, movie_id in enumerate(movies)}

        self.idx2userid = {idx: user_id for user_id, idx in self.userid2idx.items()}
        self.idx2movieid = {idx: movie_id for movie_id, idx in self.movieid2idx.items()}

        # Map userId and movieId columns to their respective indices
        self.ratings['movieId'] = ratings_df.movieId.apply(lambda x: self.movieid2idx[x])
        self.ratings['userId'] = ratings_df.userId.apply(lambda x: self.userid2idx[x])

        # Prepare input (userId, movieId) and output (rating)
        self.x = self.ratings[['userId', 'movieId']].values
        self.y = self.ratings['rating'].values

        # Convert to PyTorch tensors
        self.x = torch.tensor(self.x, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __getitem__(self, index):
        """
        Returns a single (userId, movieId) pair along with the corresponding rating.
        """
        return self.x[index], self.y[index]

    def __len__(self):
        """
        Returns the total number of ratings in the dataset.
        """
        return len(self.ratings)