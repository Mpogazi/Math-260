from math260 import data_prep, recommend, score
import numpy as np
import scipy.spatial.distance as dist


GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

def msd_similarity(u1, u2, rating_matrix, bool_matrix):
    u1_fixed = np.multiply(rating_matrix[u1], bool_matrix[u2])
    u2_fixed = np.multiply(rating_matrix[u2], bool_matrix[u1])
    diff = u1_fixed - u2_fixed
    count = bool_matrix[u2].T @ bool_matrix[u1]
    msd = (diff.T @ diff) / count
    return 1 / (msd + 1)

def cosine_similarity(u1, u2, rating_matrix, bool_matrix):
    return 1 - dist.cosine(rating_matrix[u1], rating_matrix[u2])

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix  \
        = data_prep.create_review_matrix(games, users, sparse=False, verbose=True)

    cosine_predictor = recommend.SimilarityPredictor(cosine_similarity, 120)
    msd_predictor = recommend.SimilarityPredictor(msd_similarity, 120)
    avg_predictor = recommend.AveragePredictor(rating_matrix, bool_matrix)

    cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                  cosine_predictor.predict, users=range(0, 1000))
    sim_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                               msd_predictor.predict, users=range(0, 1000))
    avg_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                               avg_predictor.predict, users=range(0, 1000))

    print(f"Cosine predictor RMSE: {cosine_rmse}")
    print(f"MSD predictor RMSE: {sim_rmse}")
    print(f"Average predictor RMSE: {avg_rmse}")


    
    



