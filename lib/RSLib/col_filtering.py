import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def import_data(ratings_file, movies_file):
    """
    Imports the data necessary for the collaborative filtering
    """
    ratings_data = pd.read_csv(ratings_file, delimiter="::", engine='python', encoding='latin1', header=None, names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies_data = pd.read_csv(movies_file, delimiter="::", engine='python', encoding='latin1', header=None, names=['MovieID', 'Title', 'Genres'])
    return ratings_data, movies_data

def create_sparse_matrix(ratings_data):
    """
    Import the matrix in a way that is efficient to store for a sparse matrix (using CSR format)
    """
    # Convert the ratings DataFrame to a sparse matrix
    ratings_data['UserID'] = pd.factorize(ratings_data['UserID'])[0]
    ratings_data['MovieID'] = pd.factorize(ratings_data['MovieID'])[0]
    
    # Then, create the sparse matrix
    sparse_matrix = csr_matrix((ratings_data['Rating'], (ratings_data['UserID'], ratings_data['MovieID'])), shape=(ratings_data['UserID'].max() + 1, ratings_data['MovieID'].max() + 1))
    
    return sparse_matrix

def compute_user_cosine_similarity(ratings_matrix):
    user_similarity = cosine_similarity(ratings_matrix)
    return user_similarity

def predict_ratings(ratings, similarity):
    weighted_ratings_sum = similarity @ ratings

    #sum_of_similarities = np.abs(similarity).sum(axis=1)

    #predicted_ratings = weighted_ratings_sum / sum_of_similarities[:, np.newaxis]

    return weighted_ratings_sum

def predict_ratings_normalize(ratings, similarity):
    weighted_ratings_sum = similarity @ ratings

    sum_of_similarities = np.abs(similarity).sum(axis=1)

    predicted_ratings = weighted_ratings_sum / sum_of_similarities[:, np.newaxis]

    return predicted_ratings


def sort_predictions_with_indices(items_predictions_arr):

    # Use numpy to get sorted indices for descending order
    sorted_indices = np.argsort(-items_predictions_arr, axis=1)

    # Use sorted indices to arrange the array in sorted order
    sorted_predictions = np.take_along_axis(items_predictions_arr, sorted_indices, axis=1)

    # Create DataFrames for the sorted values and their original indices
    df_sorted_predictions = pd.DataFrame(sorted_predictions)
    df_sorted_indices = pd.DataFrame(sorted_indices)

    return df_sorted_predictions, df_sorted_indices