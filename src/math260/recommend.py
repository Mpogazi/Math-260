"""Recommendation Algorithms Module"""

from collections import defaultdict
from enum import Enum
import numpy as np
import random

def item_similarity_recommend(users, rating_matrix, bool_matrix, similarity_function, k = -1):
    """Predicts a set of ratings for a user using
    a WMV over the k-nearest neighbors of each game.

       Args:
          users : list of user indices to recommend games for
          rating_matrix : the review matrix
          similarity_function : a function that defines the similarity between
          two sets of recommendations for a game.
          k : the number of most similar users to use in the WMV. if k == -1,
          then all users are considered.
        
       Returns:
          recommendations : a map from user indices to predicted ratings
    """
    n, m = rating_matrix.shape
    if k == -1:
        k = n 
        
    recommendations = defaultdict(list)
    for user in users:
        rec = np.zeros(m)
        weights = np.zeros(m)
        for game in range(m):
            rating = rating_matrix[user, game]
            if rating == 0:
                continue

            sim = lambda x: -1 * similarity_function(x, game, rating_matrix, bool_matrix)
            similarities = list(map(sim, range(0, m))) # TODO: make random sample
            recommenders = np.argsort(similarities)

            for recommender in recommenders[0:k]:
                weight = -1 * similarities[recommender]
                rec[recommender] += weight * rating
                weights[recommender] += weight

        weights[weights == 0] = 1
        rec = np.divide(rec, weights)
        recommendations[user] = rec

    return recommendations

def similarity_recommend(users, rating_matrix, bool_matrix, similarity_function, k = -1):
    """Predicts a set of ratings for a user using a
    weighted majority vote of their k most similar neighbors.

       Args:
          users : list of user indices to recommend games for
          rating_matrix : the review matrix
          similarity_function : a function that defines the similarity between
          two users set of recommendations. 
          k : the number of most similar users to use in the WMV. if k == -1,
          then all users are considered.
        
       Returns:
          recommendations : a map from user indices to predicted ratings
    """
    n, m = rating_matrix.shape
    if k == -1:
        k = n 
        
    recommendations = defaultdict(list)
    for user in users:
        sim = lambda x: -1 * similarity_function(x, user, rating_matrix, bool_matrix)
        similarities = list(map(sim, range(0, n))) # TODO: make random sample
        recommenders = np.argsort(similarities)

        rec = np.zeros(m)
        weights = np.zeros(m)
        for recommender in recommenders[0:k]:
            weight = -1 * similarities[recommender]
            weight_v = np.multiply(weight, bool_matrix[recommender])
            recommendation = rating_matrix[recommender]
            rec += np.multiply(weight_v, recommendation)
            weights += weight_v
        weights[weights == 0] = 1
        rec = np.divide(rec, weights)
        
        recommendations[user] = rec

    return recommendations

def random_similarity_recommend(users, rating_matrix, bool_matrix, 
                                    similarity_function, k = -1, alpha=1):
    """Predicts a set of ratings for a user using a
    weighted majority vote of their k most similar neighbors.

       Args:
          users : list of user indices to recommend games for
          rating_matrix : the review matrix
          similarity_function : a function that defines the similarity between
          two users set of recommendations. 
          k : the number of most similar users to use in the WMV. if k == -1,
          then all users are considered.
          alpha : fraction of users to randomly sample from
        
       Returns:
          recommendations : a map from user indices to predicted ratings
    """
    n, m = rating_matrix.shape
    if k == -1:
        k = n 
        
    sample = np.random.permutation(n)[:int(alpha * n)]

    recommendations = defaultdict(list)
    for user in users:
        sim = lambda x: -1 * similarity_function(x, user, rating_matrix, bool_matrix)
        similarities = list(map(sim, sample)) 
        recommenders = np.argsort(similarities)

        rec = np.zeros(m)
        weights = np.zeros(m)
        for recommender in recommenders[0:k]:
            weight = -1 * similarities[recommender]
            weight_v = np.multiply(weight, bool_matrix[recommender])
            recommendation = rating_matrix[recommender]
            rec += np.multiply(weight_v, recommendation)
            weights += weight_v
        weights[weights == 0] = 1
        rec = np.divide(rec, weights)
        
        recommendations[user] = rec

    return recommendations

class ItemSimilarityPredictor:
    """
    Implements a similarity predictor using a user similarity
    kernel. 

    This exists because the RMSE code does only one review at a time
    while my code reviews all games simultaneously and I wanted to
    cache the result.
    
    Note, SimilarityPredictor(lambda x, y: 1, -1) should be identical
    to AveragePredictor, albeit a LOT slower.

    """

    def __init__(self, similarity_f, k):
        self.recommendation = None
        self.user           = None
        self.similarity_f   = similarity_f
        self.k              = k

    def predict(self, user, game, rating_matrix, bool_matrix):
        if user == self.user:
            return self.recommendation[game]
        
        recommendations = item_similarity_recommend({user}, rating_matrix, bool_matrix,
                                                    self.similarity_f, k = self.k)
        self.user = user
        self.recommendation = recommendations[user]

        return self.recommendation[game]

class SimilarityPredictor:
    """
    Implements a similarity predictor using a user similarity
    kernel. 

    This exists because the RMSE code does only one review at a time
    while my code reviews all games simultaneously and I wanted to
    cache the result.
    
    Note, SimilarityPredictor(lambda x, y: 1, -1) should be identical
    to AveragePredictor, albeit a LOT slower.

    """

    def __init__(self, similarity_f, k):
        self.recommendation = None
        self.user           = None
        self.similarity_f   = similarity_f
        self.k              = k

    def predict(self, user, game, rating_matrix, bool_matrix):
        if user == self.user:
            return self.recommendation[game]
        
        recommendations = similarity_recommend({user}, rating_matrix, bool_matrix,
                                               self.similarity_f, k = self.k)
        self.user = user
        self.recommendation = recommendations[user]

        return self.recommendation[game]


class RandomSimilarityPredictor:

    """
    Implements the same as above, just with random sampling for the users we
    think may be nearby
    """

    def __init__(self, similarity_f, k, alpha):
        self.recommendation = None
        self.user           = None
        self.similarity_f   = similarity_f
        self.k              = k
        self.alpha          = alpha

    def predict(self, user, game, rating_matrix, bool_matrix):
        if user == self.user:
            return self.recommendation[game]
        
        recommendations = random_similarity_recommend({user}, rating_matrix, bool_matrix,
                                               self.similarity_f, k = self.k, alpha=self.alpha)
        self.user = user
        self.recommendation = recommendations[user]

        return self.recommendation[game]


class AveragePredictor:
    
    '''
    Implements an average predictor which for a given game predicts the average
    rating for that game
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

class UserAveragePredictor:
    '''
    Implements a predictor which for a given user predicts the average of their
    ratings
    '''
    
    def __init__(self):
        pass

    def predict(self, user, _game, rating_matrix, bool_matrix):
        return np.sum(rating_matrix[user,:]) / np.sum(bool_matrix[user,:])

class GameAveragePredictor:
    '''
    Implements a predictor which for a given game predicts the average of its
    ratings
    '''
    
    def __init__(self):
        pass

    def predict(self, _user, game, rating_matrix, bool_matrix):
        return np.sum(rating_matrix[:,game]) / np.sum(bool_matrix[:,game])

class GlobalAveragePredictor:
    '''
    Implements a global average predictor which always predicts the average of
    all the reviews in the dataset
    '''

    def __init__(self, rating_matrix, bool_matrix):
        self.average = np.sum(rating_matrix) / np.sum(bool_matrix)

    def predict(self, user, game, _rating_matrix, _bool_matrix):
        return self.average    

class TwoWayAveragePredictor:
    '''
    Uses the global, game, and user averages together to include bias
    from both game quality and user rating pattern
    '''

    def __init__(self, rating_matrix, bool_matrix):
        self.global_predictor = GlobalAveragePredictor(rating_matrix, bool_matrix)
        self.game_predictor = GameAveragePredictor()
        self.user_predictor = UserAveragePredictor()

    def predict(self, user, game, rating_matrix, bool_matrix):
        global_avg = self.global_predictor.predict(user, game,
            rating_matrix, bool_matrix)
        game_avg = self.game_predictor.predict(user, game,
            rating_matrix, bool_matrix)
        user_avg = self.user_predictor.predict(user, game,
            rating_matrix, bool_matrix)

        # global is the base, (user - global) adjust for harshness of user,
        # (game - global) adjust for quality of game.
        return global_avg + (user_avg - global_avg) + (game_avg - global_avg)

class RandomPredictor:
    '''
    Implements a random guesser which guesses using the random function it
    is passed
    '''

    def __init__(self, rand_func):
        self.rand_func = rand_func

    def predict(self, _user, _game, _rating_matrix, _bool_matrix):
        return self.rand_func()

class ItemPredictor:
    """
    Implements a similarity predictor using item similarity measure.
    
    """

    def __init__(self, k, rating_matrix, bool_matrix, sim_f, sim_mat_builder):
        self.sim_matrix     = sim_mat_builder(rating_matrix, bool_matrix, sim_f)
        self.sim_f          = sim_f
        self.k              = k

    def predict(self, user, game, rating_matrix, bool_matrix):
        sim_games = self.sim_matrix[game]
        ratings   = rating_matrix[user]
        x = (np.array((sim_games, ratings)))

        x = (x.T)[x[1] != 0].T
        sort = np.argsort(-x[0])
        x = x[:,sort]

        if self.k < x.shape[1]:
            ks = x[:,:self.k]
        else:
            ks = x
        prediction = (np.dot(ks[0], ks[1]) / np.sum(np.absolute(ks[0])))
        return prediction