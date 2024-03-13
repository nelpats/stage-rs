from RSLib import col_filtering
from pandas import DataFrame as df
BASE_PATH = "..\\notebooks\\datasets\\movies\\"
RATINGS_PATH = BASE_PATH + "ratings.dat"
MOVIES_PATH = BASE_PATH + "movies.dat"



def execute_col_filtering(ratings_df, movies_df):
    ratings_sparse_matrix = col_filtering.create_sparse_matrix(ratings_df)
    user_similarities = col_filtering.compute_user_cosine_similarity(ratings_sparse_matrix)
    content_prediction = col_filtering.predict_ratings_normalize(ratings_sparse_matrix, user_similarities)
    
    sorted_predictions, sorted_indices = col_filtering.sort_predictions_with_indices(content_prediction)
    return content_prediction, sorted_indices 
    
    
print("Importing datasets...")
ratings_df, movies_df = col_filtering.import_data(RATINGS_PATH, MOVIES_PATH)

print("Building training set...")
train_set, test_set = col_filtering.train_test_split_by_user(ratings_df, test_size=0.2)

print("Training set size:", len(train_set))
print("Testing set size:", len(test_set))


print("Making predictions...")
pred, indices = execute_col_filtering(train_set, movies_df)
scaled_pred = df(col_filtering.scale_predictions_to_ratings_range(pred))


rmse = col_filtering.evaluate_predictions_rmse(scaled_pred, test_set)

print(f"{rmse=}")