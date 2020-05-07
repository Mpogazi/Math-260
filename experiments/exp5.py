from math260 import data_prep, recommend, score, item_based_cf
import numpy as np
import scipy.spatial.distance as dist

# PLEASE SWAP THESE FOR REAL EVALUATION
# "data/2019-05-02-debug.csv" # "data/games.csv" #
# "data/bgg-13m-reviews-debug.csv" # "data/reviews.csv" #
GAMES_FILE = "data/2019-05-02-debug.csv" 
REVIEWS_FILE =  "data/bgg-13m-reviews-debug.csv" 

# This is e^-(1/n |u - v|_1) which we use as our kernel here
# it doesn't work great but oh well

def item_cb_similarity(ratings, bools):
    overlap = np.multiply(bools[:,0], bools[:,1])
    return  np.exp(-dist.cityblock(
            ratings[:,0], ratings[:,1], w=overlap
        ) / np.sum(overlap))
    
if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix = \
        data_prep.create_review_matrix(games, users, sparse=False, verbose=True)
    
    removed = score.remove_fraction(0.1,rating_matrix, bool_matrix)

    cb_predictor = recommend.ItemPredictor(100, rating_matrix, bool_matrix, 
                    item_cb_similarity, item_based_cf.sim_matrix)

    cb_rmse = score.rmse(removed, rating_matrix, bool_matrix, 
                    cb_predictor.predict, users=range(0, 1000))

    print(f"Item-based City-block predictor RMSE: {cb_rmse}")