from math260 import data_prep, recommend, score
import numpy as np
import scipy.spatial.distance as dist


GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

L = 1

def msd_similarity(u1, u2, rating_matrix, bool_matrix):
    u1_fixed = np.multiply(rating_matrix[u1], bool_matrix[u2])
    u2_fixed = np.multiply(rating_matrix[u2], bool_matrix[u1])
    diff = u1_fixed - u2_fixed
    count = bool_matrix[u2].T @ bool_matrix[u1]
    msd = (diff.T @ diff) / count
    return L / (msd + L)

def adjusted_cosine_similarity(u1, u2, rating_matrix, bool_matrix):
    r1 = rating_matrix[u1] 
    r2 = rating_matrix[u2]
    r1 = r1 - np.mean(r1)
    r2 = r1 - np.mean(r2)
    num = r1.T @ r2
    den = np.sqrt((r1.T @ r1) * (r2 @ r2))
    if den == 0:
        den = 1
    return num / den

def cosine_similarity(u1, u2, rating_matrix, bool_matrix):
    r1 = rating_matrix[u1]
    r2 = rating_matrix[u2]
    num = r1.T @ r2
    den = np.sqrt((r1.T @ r1) * (r2 @ r2))
    if den == 0:
        den = 1
    return num / den

def jaccard_similarity(u1, u2, rating_matrix, bool_matrix):
    inter = bool_matrix[u1].T @ bool_matrix[u2]
    union = bool_matrix[u1].T @ bool_matrix[u1] + bool_matrix[u2].T @ bool_matrix[u2]
    return inter / union

def combo_sim(u1, u2, rating_matrix, bool_matrix):
    return jaccard_similarity(u1, u2, rating_matrix, bool_matrix) * msd_similarity(u1, u2, rating_matrix, bool_matrix)

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix  \
        = data_prep.create_review_matrix(games, users, sparse=False, verbose=True)

    combo_predictor = recommend.SimilarityPredictor(combo_sim, 120)
    adjusted_cosine_predictor = recommend.SimilarityPredictor(adjusted_cosine_similarity, 120)
    cosine_predictor = recommend.SimilarityPredictor(cosine_similarity, 120)
    msd_predictor = recommend.SimilarityPredictor(msd_similarity, 120)
    jaccard_predictor = recommend.SimilarityPredictor(jaccard_similarity, 120)
    avg_predictor = recommend.AveragePredictor(rating_matrix, bool_matrix)
    uavg_predictor = recommend.AveragePredictor(rating_matrix, bool_matrix)
    gavg_predictor = recommend.GlobalAveragePredictor(rating_matrix, bool_matrix)
    tavg_predictor = recommend.TwoWayAveragePredictor(rating_matrix, bool_matrix)

    ## adjusted_cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
    ##                                        adjusted_cosine_predictor.predict, users=range(0, 1000))
    ## cosine_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
    ##                               cosine_predictor.predict, users=range(0, 1000))
    combo_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                 combo_predictor.predict, users=range(0, 1000))
    jaccard_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                   jaccard_predictor.predict, users=range(0, 1000))
    sim_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                               msd_predictor.predict, users=range(0, 1000))
    avg_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                               avg_predictor.predict, users=range(0, 1000))
    uavg_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                uavg_predictor.predict, users=range(0, 1000))
    gavg_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                gavg_predictor.predict, users=range(0, 1000))
    tavg_rmse, _ = score.rmsecv(0.1, rating_matrix, bool_matrix,
                                tavg_predictor.predict, users=range(0, 1000))

    # print(f"Adjusted cosine predictor RMSE: {adjusted_cosine_rmse}")
    # print(f"Cosine predictor RMSE: {cosine_rmse}")
    print(f"Combo predictor RMSE: {combo_rmse}")
    print(f"MSD predictor RMSE: {sim_rmse}")
    print(f"Jaccard predictor RMSE: {jaccard_rmse}")
    print(f"Game average predictor RMSE: {avg_rmse}")
    print(f"User average predictor RMSE: {uavg_rmse}")
    print(f"Global average predictor RMSE: {gavg_rmse}")
    print(f"Two way average predictor RMSE: {tavg_rmse}")



    
    



