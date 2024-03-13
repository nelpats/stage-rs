from RSLib import col_filtering

BASE_PATH = "..\\notebooks\\datasets\\movies\\"
RATINGS_PATH = BASE_PATH + "ratings.dat"
MOVIES_PATH = BASE_PATH + "movies.dat"

ratings_df, movies_df = col_filtering.import_data(RATINGS_PATH, MOVIES_PATH)

ratings_sparse_matrix = col_filtering.create_sparse_matrix(ratings_df)
user_similarities = col_filtering.compute_user_cosine_similarity(ratings_sparse_matrix)

content_prediction = col_filtering.predict_ratings(ratings_sparse_matrix, user_similarities)
sorted_content_prediction = col_filtering.sort_predictions_with_indices(content_prediction)

print(sorted_content_prediction)
