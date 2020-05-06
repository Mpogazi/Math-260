from math260 import data_prep, recommend, score, item_based_cf
import numpy as np

# PLEASE SWAP THESE FOR REAL EVALUATION

GAMES_FILE = "data/2019-05-02-debug.csv" #"data/games.csv"
REVIEWS_FILE = "data/bgg-13m-reviews-debug.csv" #"data/reviews.csv"

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix = data_prep.create_review_matrix(games, users, sparse=False, verbose=False)
    #cosine_predictor = recommend.ItemPredictor(cos_matrix, 120, games_map)
    pearson_predictor = recommend.ItemPredictor(10, games_map, item_based_cf.PearsonSimilarity)

    #cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix, cosine_predictor.predict, users=range(0, 1000))
    pearson_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                               pearson_predictor.predict, users=range(0, 1000))

    #print(f"Item-based Cosine predictor RMSE: {cosine_rmse}")
    print(f"Item-based Pearson predictor RMSE: {pearson_rmse}")