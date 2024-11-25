import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import networkx as nx
from matrix_factorization import MatrixFactorization
from ui import start_ui
from data_loader import Loader

movies_df = pd.read_csv('movies.csv').sample(1000)
ratings_df = pd.read_csv('ratings.csv').sample(1000)

movies_df = pd.DataFrame({
    'movieId': [1, 2, 3],
    'title': ['Movie 1', 'Movie 2', 'Movie 3'],
    'genres': ['Action', 'Comedy', 'Drama']
})

# Graph creation
G = nx.Graph()
for _, row in ratings_df.iterrows():
    G.add_edge(f"user_{row['userId']}", f"movie_{row['movieId']}", weight=row['rating'])

# Collaborative filtering setup
n_users = len(ratings_df.userId.unique())
n_items = len(movies_df.movieId.unique())
model = MatrixFactorization(n_users, n_items, n_factors=8)
cuda = torch.cuda.is_available()
cuda = False
if cuda:
    model = model.cuda()

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 5

# Train the model
for it in range(num_epochs):
    print(f"Epoch {it + 1}/{num_epochs}")

    losses = []
    for x, y in DataLoader(Loader(ratings_df), batch_size=512, shuffle=True):
        if cuda:
            x, y = x.cuda