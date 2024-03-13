import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.metrics import mean_squared_error
from math import sqrt




def evaluate_predictions_rmse(predictions, actual_ratings):
    """
    Evaluate the accuracy of the predictions using RMSE.
    
    Parameters:
        predictions (pd.DataFrame): A dataframe with 'UserID', 'MovieID', and 'PredictedRating'.
        actual_ratings (pd.DataFrame): The test set with actual 'UserID', 'MovieID', and 'Rating'.
        
    Returns:
        float: The RMSE of the predictions.
    """
    # Merge the predictions with the actual ratings to align them
    comparison_df = predictions.merge(actual_ratings, on=['UserID', 'MovieID'])
    
    # Compute RMSE
    mse = mean_squared_error(comparison_df['Rating'], comparison_df['PredictedRating'])
    rmse = sqrt(mse)
    
    return rmse

def train_test_split_by_user(ratings_data, test_size=0.2):
    """
    Splits the ratings data into training and testing sets, ensuring each user appears in both sets.
    
    Parameters:
        ratings_data (pd.DataFrame): The full ratings dataset.
        test_size (float): The proportion of ratings per user to include in the test set.
        
    Returns:
        train_set (pd.DataFrame): The training set.
        test_set (pd.DataFrame): The testing set.
    """
    # Ensure the test size is within a reasonable range
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    
    # Shuffle the data
    shuffled_data = ratings_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Initialize empty dataframes for the training and testing sets
    train_set = pd.DataFrame()
    test_set = pd.DataFrame()
    
    # Group by UserID and split for each user
    for _, group in shuffled_data.groupby('UserID'):
        # Calculate the number of ratings to use as test data for this user
        num_test_ratings = max(1, int(np.floor(test_size * len(group))))
        
        test_ratings = group.iloc[:num_test_ratings]
        train_ratings = group.iloc[num_test_ratings:]
        
        test_set = pd.concat([test_set, test_ratings])
        train_set = pd.concat([train_set, train_ratings])
    
    return train_set, test_set


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

def scale_predictions_to_ratings_range(predicted_ratings, min_rating=1, max_rating=5):
    """
    Scales the predicted ratings to a specified ratings range using the Min-Max normalization
    """
    min_pred = np.min(predicted_ratings)
    max_pred = np.max(predicted_ratings)
    scaled_ratings = min_rating + (predicted_ratings - min_pred) * (max_rating - min_rating) / (max_pred - min_pred)
    return scaled_ratings