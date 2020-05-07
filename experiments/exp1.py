from math260 import data_prep, recommend, score
import numpy as np

GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix  \
        = data_prep.create_review_matrix(games, users, sparse=False, verbose=True)

    game_predictor = recommend.AveragePredictor(np.copy(rating_matrix), np.copy(bool_matrix))
    removed = score.remove_fraction(0.1,rating_matrix, bool_matrix)

    # testing removing 10% from each user and predicting using average score
    global_predictor = recommend.GlobalAveragePredictor(rating_matrix, bool_matrix)
    user_predictor = recommend.UserAveragePredictor()
    tw_predictor = recommend.TwoWayAveragePredictor(rating_matrix, bool_matrix)


    glob_avg_rmse = score.rmse(removed, rating_matrix, bool_matrix, 
                    global_predictor.predict, users=range(0, 1000))
    game_avg_rmse = score.rmse(removed, rating_matrix, bool_matrix, 
                    game_predictor.predict, users=range(0, 1000))
    user_avg_rmse = score.rmse(removed, rating_matrix, bool_matrix, 
                    user_predictor.predict, users=range(0, 1000))
    tw_avg_rmse = score.rmse(removed, rating_matrix, bool_matrix, 
                    tw_predictor.predict, users=range(0, 1000))

    print('Global:\t\t{}'.format(glob_avg_rmse))
    print('Game:\t\t{}'.format(game_avg_rmse))
    print('User:\t\t{}'.format(user_avg_rmse))
    print('Two-way:\t\t{}'.format(tw_avg_rmse))