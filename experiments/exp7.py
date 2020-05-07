from math260 import data_prep, recommend, score
import numpy as np
import scipy.spatial.distance as dist
import json


GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

output = 'figures/random-sample.txt'

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

    results = []

    for i in range(1,11):
        alpha = i / 10

        #cosine_predictor = recommend.RandomSimilarityPredictor(cosine_similarity, 120, alpha)
        msd_predictor = recommend.RandomSimilarityPredictor(msd_similarity, 120, alpha)

        #cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
        #                        cosine_predictor.predict, users=range(0, 1000))
        sim_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                msd_predictor.predict, users=range(0, 1000))

        result = {'alpha': alpha, 'sim':sim_rmse } #, 'cos':cosine_rmse}
        print(result)
        results.append(result)

    json.dump(results, open(output, mode='w'))
