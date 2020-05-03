from collections import defaultdict
import numpy as np
import random 

def random_recommend(users, review_matrix, seed=None):
    """Recommends a random game that is not yet played for all users.
       
       Args: 
          users : list of user indices to recommend games for
          review_matrix : the review matrix
          seed: an optional seed for initializing randomness

       Returns:
          recommendations : a map from user indices to a game indices
    """
    random.seed(seed)

    recommendations, num_games = defaultdict(list), review_matrix.shape[1]
    for user in users:
        games = list(range(0, num_games))
        random.shuffle(games)
        for game in games:
            if review_matrix[user, game] == 0:
                recommendations[user].append(game)
    return recommendations
                
def popular_recommend(users, review_matrix, ranked_games):
    """Recommends the most popular game not yet played for all
       users.
       
       Args: 
          users : list of user indices to recommend games for
          review_matrix : the review matrix
          ranked_games : a list of games indices ranked from best to worst

       Returns:
          recommendations : a map from user indices to game indices
    """
    recommendations = defaultdict(list)
    for user in users:
        for game in ranked_games:
            if review_matrix[user, game] == 0:
                recommendations[user].append(game)
    return recommendations
