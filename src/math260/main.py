import data_prep
import recommend

GAMES_DATA_FILE = "data/2019-05-02-debug.csv"
# This must be the debug dataset if working on a personal machine.
# The full dataset is 2/3 comments which take up way too much memory
REVIEWS_FILE = "data/bgg-13m-reviews-debug.csv"

if __name__ == "__main__":
    print("Hello friends :), we start here!")


'''
loads the data about the users into two maps, which are corelated by the
reviews

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

'''
creates two maps from {game, user} name to index, a boolean matrix of if the
user has reviewed that game, and the actual matrix with reviews as entries 
'''
games_map, users_map, review_matrix, bool_matrix = data_prep.create_review_matrix(
    games, users, True)
