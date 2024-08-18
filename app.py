import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
final_dataset = ratings.pivot(index='movieId', columns='userId', values='rating')
final_dataset.fillna(0, inplace=True)

# Create user and movie vote counts
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')

# Filter out movies with less than 10 votes
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]

# Filter out users with less than 50 votes
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]

# Calculate sparsity
sample = np.array([[0, 0, 3, 0, 0], [4, 0, 0, 0, 2], [0, 0, 0, 0, 1]])
sparsity = 1.0 - (np.count_nonzero(sample) / float(sample.size))
print(sparsity)

# Create sparse matrix
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)

# Fit KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Function to get movie recommendations
def get_movie_recommendation(movie_name):
    n_movies_to_reccomend = 10
    movie_list = movies[movies['title'].str.contains(movie_name)]
    if len(movie_list):
        movie_idx = movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]

        distances, indices = knn.kneighbors(csr_data[movie_idx], n_neighbors=n_movies_to_reccomend + 1)
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())),
                                   key=lambda x: x[1])[:0:-1]

        recommend_frame = []

        for val in rec_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title': movies.iloc[idx]['title'].values[0], 'Distance': val[1]})
        df = pd.DataFrame(recommend_frame, index=range(1, n_movies_to_reccomend + 1))
        return df

    else:
        return "No movies found. Please check your input"

# Streamlit UI
st.set_page_config(page_title="Movie Recommendation System", page_icon="🎥", layout="centered", initial_sidebar_state="auto")

st.title("🎥 Movie Recommendation System")
st.markdown("Find movie recommendations based on your favorite movies!")

# Input movie name
movie_name = st.text_input("Enter a movie name", "")

# Display recommendations
if st.button('Get Recommendations'):
    if movie_name:
        recommendations = get_movie_recommendation(movie_name)
        if isinstance(recommendations, pd.DataFrame):
            st.subheader("Top 10 Recommended Movies")
            st.dataframe(recommendations)
        else:
            st.error(recommendations)
    else:
        st.warning("Please enter a movie name.")

# Apply CSS for custom styling
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
        color: #000;
    }
    .title {
        color: #b22222;
    }
    </style>
    """,
    unsafe_allow_html=True
)
