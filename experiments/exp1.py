from math260 import data_prep, recommend, score

GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, bool_matrix  \
        = data_prep.create_review_matrix(games, users, sparse=False, verbose=True)

    # testing removing 10% from each user and predicting using average score
    predictor = recommend.AveragePredictor(rating_matrix, bool_matrix)

    rmse, errors = score.rmsecv(0.1, rating_matrix, bool_matrix, predictor.predict)

    print('RMSE:\t\t{}'.format(rmse))