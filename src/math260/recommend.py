"""Recommendation Algorithms Module

All functions in this module recommend a ranked list of games from
most likely to least likely for a set of users based upon the review
matrix.

The review matrix is expected to have rows represent users and columns
represent games. An entry represents whether a user likes a game or
not (1/0 respectively).

"""

from collections import defaultdict
from enum import Enum
import numpy as np
import random

def random_recommend(users, review_matrix, seed=None):
    """Recommends random games that is not yet played for all users.
       
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
    """Recommends the most popular games not yet played for all
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

def similarity_recommend(users, review_matrix, similarity_function, k = -1):
    """Recommends a set of games that have not been rated already using a
    weighted majority vote of their k most similar neighbors.

       Args:
          users : list of user indices to recommend games for
          review_matrix : the review matrix
          similarity_function : a function that defines the similarity between
          two users set of recommendations. requires that users are most similar to
          themselves.
          k : the number of most similar users to use in the WMV. if k == -1,
          then all users are considered.
        
       Returns:
          recommendations : a map from user indices to game indices
    """
    n = review_matrix.shape[1]
    if k == -1:
        k = n - 1
        
    recommendations = defaultdict(list)
    for user in users:
        similarities = list(map(lambda x: -1 * similarity_function(review_matrix[user], x), review_matrix))
        recommenders = np.argsort(similarities) # a list of most similar users sorted from most to least similar

        rec = np.zeros(n)
        for recommender in recommenders[1:k + 1]:
            weight = similarities[recommender]
            recommendation = review_matrix[recommender]
            rec += -1 * weight * recommendation
        
        recommenders[user] = np.argsort(rec)

    return recommendations

class AveragePredictor:
    
    '''
    Implements an average predictor which is sped up by using online average
    calculations.
    '''

    def __init__(self, rating_matrix, bool_matrix):
        self.rating_matrix = np.copy(rating_matrix)
        self.bool_matrix = np.copy(bool_matrix)
        self.rating_sum = np.sum(rating_matrix, axis=0)
        self.counts = np.sum(bool_matrix, axis=0)

    def predict(self, user, game, _rating_matrix, _bool_matrix):
        game_sum = self.rating_sum[game]
        game_count = self.counts[game]
        if self.bool_matrix[user, game] == 1:
            return (game_sum - self.rating_matrix[user, game]) / (game_count - 1)
        else:
            return game_sum / game_count

class RandomPredictor:
    
    '''
    Implements a random guesser which guesses using the random function it
    is passed
    '''

    def __init__(self, rand_func):
        self.rand_func = rand_func

    def predict(self, _user, _game, _rating_matrix, _bool_matrix):
        return self.rand_func()
