import data_prep

GAMES_DATA_FILE = "data/2019-05-02.csv"
# This must be the minimized dataset if working on a personal machine.
# The full dataset is 2/3 comments which take up way too much memory
REVIEWS_FILE = "data/bgg-13m-reviews-min.csv"

if __name__ == "__main__":
    print("Hello friends :), we start here!")


'''
games - {
    game_name: {
        name: game_name,
        reviews: [Review],
        users: [User],
        users_dict: {user_name: User},
        ...
    }
}
users - {
    user_name: {
        name: user_name
        reviews: [Review],
        games: [Game],
        games_dict: {game_name: Game}
    }
}
'''
games, users = data_prep.parse_data(
    GAMES_DATA_FILE, REVIEWS_FILE, True)
