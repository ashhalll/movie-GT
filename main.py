import torch
import pandas as pd
import numpy as np
import networkx as nx
import customtkinter as ctk
import matplotlib.pyplot as plt
from networkx.drawing.layout import bipartite_layout
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

# Matrix Factorization Model with enhanced collaborative filtering
class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=32):  # Increased factors
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)
        self.user_bias = torch.nn.Embedding(n_users, 1)
        self.item_bias = torch.nn.Embedding(n_items, 1)
        self.global_bias = torch.nn.Parameter(torch.zeros(1))

        # Initialize weights with higher variance
        torch.nn.init.normal_(self.user_factors.weight, std=0.1)
        torch.nn.init.normal_(self.item_factors.weight, std=0.1)
        torch.nn.init.normal_(self.user_bias.weight, std=0.05)
        torch.nn.init.normal_(self.item_bias.weight, std=0.05)

    def forward(self, data):
        users, items = data[:, 0], data[:, 1]
        dot = (self.user_factors(users) * self.item_factors(items)).sum(1)
        pred = dot + self.user_bias(users).squeeze() + self.item_bias(items).squeeze() + self.global_bias
        return pred

# Enhanced Loader Class with improved similarity computation
class Loader:
    def __init__(self, ratings_df):
        self.userid2idx = {user_id: idx for idx, user_id in enumerate(ratings_df['userId'].unique())}
        self.movieid2idx = {movie_id: idx for idx, movie_id in enumerate(ratings_df['movieId'].unique())}
        self.idx2userid = {idx: user_id for user_id, idx in self.userid2idx.items()}
        self.idx2movieid = {idx: movie_id for movie_id, idx in self.movieid2idx.items()}

        # Create normalized user-item matrix for better similarity calculations
        self.user_item_matrix = self._create_user_item_matrix(ratings_df)
        # Normalize ratings per user
        user_means = np.nanmean(self.user_item_matrix, axis=1, keepdims=True)
        user_stds = np.nanstd(self.user_item_matrix, axis=1, keepdims=True)
        user_stds[user_stds == 0] = 1  # Avoid division by zero
        self.normalized_matrix = (self.user_item_matrix - user_means) / user_stds
        self.normalized_matrix = np.nan_to_num(self.normalized_matrix)

        # Compute similarity with normalized ratings
        self.user_similarity = cosine_similarity(self.normalized_matrix)

        # Map userId and movieId to indices
        ratings_df['user_index'] = ratings_df['userId'].map(self.userid2idx)
        ratings_df['movie_index'] = ratings_df['movieId'].map(self.movieid2idx)
        self.ratings = ratings_df[['user_index', 'movie_index', 'rating']].to_numpy()

    def _create_user_item_matrix(self, ratings_df):
        matrix = np.full((len(self.userid2idx), len(self.movieid2idx)), np.nan)
        for _, row in ratings_df.iterrows():
            user_idx = self.userid2idx[row['userId']]
            movie_idx = self.movieid2idx[row['movieId']]
            matrix[user_idx, movie_idx] = row['rating']
        return matrix

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.ratings[idx, :2], self.ratings[idx, 2]

# Load data
movies_df = pd.read_csv('movies.csv')  # Format: movieId,title,genres
ratings_df = pd.read_csv('ratings.csv')  # Format: userId,movieId,rating,timestamp

# Enhanced bipartite graph creation with weighted edges
def create_bipartite_graph(ratings_df):
    G = nx.Graph()

    # Add nodes with attributes
    users = ratings_df['userId'].unique()
    movies = ratings_df['movieId'].unique()

    for user in users:
        G.add_node(f"user_{user}", bipartite=0, node_type='user')
    for movie in movies:
        G.add_node(f"movie_{movie}", bipartite=1, node_type='movie')

    # Add weighted edges with rating information
    for _, row in ratings_df.iterrows():
        G.add_edge(
            f"user_{row['userId']}",
            f"movie_{row['movieId']}",
            weight=row['rating'],
            timestamp=row['timestamp']
        )
    return G

# Improved Collaborative Filtering Setup
def train_model(ratings_df, n_users, n_items):
    model = MatrixFactorization(n_users, n_items, n_factors=32)  # Increased factors
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.02)  # Adjusted parameters
    cuda = torch.cuda.is_available()

    if cuda:
        model = model.cuda()

    num_epochs = 20  # Increased epochs
    batch_size = 512  # Adjusted batch size
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        losses = []
        for x, y in DataLoader(Loader(ratings_df), batch_size=batch_size, shuffle=True):
            if cuda:
                x, y = x.cuda(), y.cuda()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        epoch_loss = np.mean(losses)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss

    return model

# Enhanced Recommendation Functionality with diversity
def recommend_movies(user_id, top_n=5, train_set=None, full_matrix=None):
    user_index = train_set.userid2idx.get(user_id, None)
    if user_index is None:
        return "User ID not found."

    # Get user ratings and similar users
    user_ratings = train_set.normalized_matrix[user_index]
    similar_users = train_set.user_similarity[user_index]

    # Get more similar users for diversity
    similar_user_indices = np.argsort(similar_users)[-10:]  # Increased to top 10 similar users

    # Weighted prediction using similar users with diversity factor
    weighted_predictions = np.zeros_like(user_ratings)
    similarity_sum = 0

    for i, sim_user_idx in enumerate(similar_user_indices):
        # Add diversity weight that decreases with similarity rank
        diversity_weight = 1.0 / (i + 1)
        weighted_predictions += (similar_users[sim_user_idx] * diversity_weight *
                               train_set.normalized_matrix[sim_user_idx])
        similarity_sum += similar_users[sim_user_idx] * diversity_weight

    weighted_predictions /= similarity_sum

    # Add genre diversity bonus
    genre_bonus = np.zeros_like(weighted_predictions)
    user_genres = set()
    for movie_idx in np.where(~np.isnan(full_matrix[user_index]))[0]:
        movie_id = train_set.idx2movieid[movie_idx]
        genres = movies_df[movies_df['movieId'] == movie_id]['genres'].iloc[0].split('|')
        user_genres.update(genres)

    for movie_idx in range(len(genre_bonus)):
        if movie_idx in train_set.idx2movieid:
            movie_id = train_set.idx2movieid[movie_idx]
            movie_genres = movies_df[movies_df['movieId'] == movie_id]['genres'].iloc[0].split('|')
            # Bonus for movies with new genres
            new_genres = set(movie_genres) - user_genres
            genre_bonus[movie_idx] = len(new_genres) * 0.1

    weighted_predictions += genre_bonus

    # Remove already rated movies
    rated_movie_indices = ratings_df[ratings_df['userId'] == user_id]['movieId'].map(
        lambda x: train_set.movieid2idx.get(x, -1)
    ).dropna().astype(int).tolist()

    for idx in rated_movie_indices:
        weighted_predictions[idx] = -np.inf

    # Get top recommendations
    top_movie_indices = weighted_predictions.argsort()[-top_n:][::-1]
    top_movie_ids = [train_set.idx2movieid[idx] for idx in top_movie_indices]

    recommendations = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title', 'genres']]
    return recommendations

# Enhanced GUI Setup
def start_ui(graph, train_set, full_matrix):
    app = ctk.CTk()
    app.geometry('1000x800')
    app.title('Advanced Movie Recommendation System')

    # Set color theme
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    def get_recommendations():
        try:
            user_id = int(entry_user_id.get())
            top_n = int(entry_top_n.get())
            recommendations = recommend_movies(user_id, top_n, train_set, full_matrix)

            # Clear previous results
            result_text.delete('1.0', ctk.END)

            # Format recommendations nicely
            result_text.insert(ctk.END, "Recommended Movies:\n\n", 'header')
            for _, row in recommendations.iterrows():
                result_text.insert(ctk.END, f"Title: {row['title']}\n", 'title')
                result_text.insert(ctk.END, f"Genres: {row['genres']}\n", 'genres')
                result_text.insert(ctk.END, "-" * 50 + "\n\n")

        except ValueError:
            result_text.delete('1.0', ctk.END)
            result_text.insert(ctk.END, "Please enter valid numeric values.")

    def show_bipartite_graph():
        plt.style.use('dark_background')
        plt.figure(figsize=(15, 10), dpi=300, facecolor='#1a1a1a')

        users = [n for n in graph if n.startswith("user_")]
        movies = [n for n in graph if n.startswith("movie_")]

        pos = bipartite_layout(graph, users, align='horizontal')
        edge_weights = [graph[u][v]['weight']/5.0 for (u,v) in graph.edges()]

        # Enhanced visualization
        nx.draw_networkx_edges(
            graph, pos,
            alpha=0.2,
            edge_color='#4a88ff',
            width=[w*0.8 for w in edge_weights]
        )

        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=users,
            node_color='#ff6b6b',
            node_size=1000,
            alpha=0.9,
            node_shape='o'
        )

        nx.draw_networkx_nodes(
            graph, pos,
            nodelist=movies,
            node_color='#4ecdc4',
            node_size=1000,
            alpha=0.9,
            node_shape='s'
        )

        labels = {node: node.split('_')[1] for node in graph.nodes()}
        nx.draw_networkx_labels(
            graph, pos,
            labels,
            font_size=8,
            font_color='white',
            font_weight='bold',
            font_family='sans-serif'
        )

        plt.title("User-Movie Interaction Network",
                 fontsize=20,
                 color='white',
                 pad=20,
                 fontweight='bold',
                 fontfamily='sans-serif')

        plt.plot([], [], 'o', color='#ff6b6b', label='Users', markersize=10)
        plt.plot([], [], 's', color='#4ecdc4', label='Movies', markersize=10)
        plt.legend(fontsize=12,
                  loc='upper right',
                  facecolor='#1a1a1a',
                  edgecolor='none',
                  labelcolor='white')

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Main content frame
    main_frame = ctk.CTkFrame(master=app)
    main_frame.pack(pady=20, padx=20, fill='both', expand=True)

    # Title
    title_label = ctk.CTkLabel(
        main_frame,
        text="Movie Recommendation System",
        font=("Helvetica", 24, "bold")
    )
    title_label.pack(pady=20)

    # Input frame
    input_frame = ctk.CTkFrame(main_frame)
    input_frame.pack(pady=10, padx=20, fill='x')

    # User ID input
    user_frame = ctk.CTkFrame(input_frame)
    user_frame.pack(side='left', padx=10, expand=True)

    ctk.CTkLabel(user_frame, text="User ID:").pack()
    entry_user_id = ctk.CTkEntry(user_frame, placeholder_text="Enter User ID")
    entry_user_id.pack(pady=5)

    # Top N input
    top_n_frame = ctk.CTkFrame(input_frame)
    top_n_frame.pack(side='left', padx=10, expand=True)

    ctk.CTkLabel(top_n_frame, text="Number of Recommendations:").pack()
    entry_top_n = ctk.CTkEntry(top_n_frame, placeholder_text="Enter number")
    entry_top_n.pack(pady=5)

    # Buttons frame
    button_frame = ctk.CTkFrame(main_frame)
    button_frame.pack(pady=20)

    # Action buttons
    ctk.CTkButton(
        button_frame,
        text="Get Recommendations",
        command=get_recommendations,
        width=200,
        height=40
    ).pack(side='left', padx=10)

    ctk.CTkButton(
        button_frame,
        text="View Interaction Network",
        command=show_bipartite_graph,
        width=200,
        height=40
    ).pack(side='left', padx=10)

    # Results area
    result_frame = ctk.CTkFrame(main_frame)
    result_frame.pack(pady=20, padx=20, fill='both', expand=True)

    result_text = ctk.CTkTextbox(
        result_frame,
        wrap='word',
        height=300,
        font=("Helvetica", 12)
    )
    result_text.pack(pady=10, padx=10, fill='both', expand=True)

    # Configure tags for text formatting
    result_text.tag_config('header', font=("Helvetica", 14, "bold"))
    result_text.tag_config('title', font=("Helvetica", 12, "bold"))
    result_text.tag_config('genres', font=("Helvetica", 10))

    app.mainloop()

# Main Functionality
if __name__ == "__main__":
    # Initialize Loader for data mappings
    train_set = Loader(ratings_df)

    # Train the model and get predictions
    n_users = len(train_set.userid2idx)
    n_items = len(train_set.movieid2idx)
    model = train_model(ratings_df, n_users, n_items)

    # Generate full prediction matrix
    full_matrix = train_set.user_item_matrix.copy()

    # Create bipartite graph
    graph = create_bipartite_graph(ratings_df)

    # Start the enhanced UI
    start_ui(graph, train_set, full_matrix)