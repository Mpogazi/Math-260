from math260 import data_prep, recommend, score

GAMES_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

if __name__ == "__main__":
    games, users = data_prep.parse_data(GAMES_FILE, REVIEWS_FILE, verbose=True)
    games_map, users_map, rating_matrix, _ = data_prep.create_review_matrix(games, users, verbose=True)
