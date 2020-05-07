from math260 import data_prep, recommend, score, item_based_cf
import numpy as np


GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=False)
    games_map, users_map, rating_matrix, bool_matrix = data_prep.create_review_matrix(games, users, sparse=False, verbose=True)
    

    cosine_predictor = recommend.ItemPredictor(120, rating_matrix, item_based_cf.CosineSimilarity, item_based_cf.sim_matrix)
    pearson_predictor= recommend.ItemPredictor(120, rating_matrix, item_based_cf.PearsonSimilarity, item_based_cf.sim_matrix)

    cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix, cosine_predictor.predict, users=range(0, 1000))
    pearson_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix, pearson_predictor.predict, users=range(0, 1000))

    print(f"Item-based Cosine predictor RMSE: {cosine_rmse}")
    print(f"Item-based Pearson predictor RMSE: {pearson_rmse}")