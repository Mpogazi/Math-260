from math260 import data_prep, recommend, score
import numpy as np
import scipy.spatial.distance as dist

GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

def item_cosine_similarity(g1, g2, rating_matrix, bool_matrix):
    return 1 - dist.cosine(rating_matrix[:, g1], rating_matrix[:, g2])

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix  \
        = data_prep.create_review_matrix(games, users, sparse=False, verbose=True)

    cosine_predictor = recommend.ItemSimilarityPredictor(item_cosine_similarity, 120, rating_matrix, bool_matrix)
    cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                  cosine_predictor.predict, users=range(0, 1000))
    print(f"Cosine predictor RMSE: {cosine_rmse}")


    
    



