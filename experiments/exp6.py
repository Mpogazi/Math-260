from math260 import data_prep, recommend, score
import numpy as np
import scipy.spatial.distance as dist

GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

def item_cosine_similarity(g1, g2, rating_matrix, bool_matrix):
    r1 = rating_matrix[:, g1]
    r2 = rating_matrix[:, g2]
    num = r1.T @ r2
    den = np.sqrt((r1.T @ r1) * (r2 @ r2))
    if den == 0:
        den = 1
    return num / den

def item_jaccard_similarity(g1, g2, rating_matrix, bool_matrix):
    inter = bool_matrix[:, g1].T @ bool_matrix[:, g2]
    union = bool_matrix[:, g1].T @ bool_matrix[:, g1] + bool_matrix[:, g2].T @ bool_matrix[:, g2]
    return inter / union

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix  \
        = data_prep.create_review_matrix(games, users, sparse=False, verbose=True)

    cosine_predictor = recommend.ItemSimilarityPredictor(item_jaccard_similarity, 10, rating_matrix, bool_matrix)
    cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                  cosine_predictor.predict, users=range(0, 1000))
    jaccard_predictor = recommend.ItemSimilarityPredictor(item_jaccard_similarity, 10, rating_matrix, bool_matrix)
    jaccard_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                  jaccard_predictor.predict, users=range(0, 1000))
    print(f"Cosine predictor RMSE: {cosine_rmse}")
    print(f"Jaccard predictor RMSE: {jaccard_rmse}")


    
    



