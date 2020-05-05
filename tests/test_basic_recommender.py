import unittest
import numpy as np
from math260 import score, recommend, data_prep

# Example of 6 users and 4 games.
# Note that it is binary
review_matrix = np.array([[0, 1, 1, 0],
                          [1, 0, 1, 1],
                          [0, 0, 1, 1],
                          [0, 0, 0, 1],
                          [1, 1, 1, 0],
                          [0, 0, 1, 0]])

# ranks games from most to least popular (breaks ties lexographically)
ranked_games = np.argsort(-np.sum(review_matrix, axis=0))

class TestRecommenders(unittest.TestCase):
    def test_popular_recommend(self):
        recommendations = recommend.popular_recommend(range(0, 6), review_matrix, ranked_games)
        correct = {0: [3, 0], 1: [1], 2: [0, 1], 3: [2, 0, 1], 4: [3], 5: [3, 0, 1]}
        for user, games in correct.items():
            self.assertEqual(games, recommendations[user])

    def test_random_recommend(self):
        recommendations = recommend.random_recommend(range(0, 6), review_matrix, seed=10)
        correct = {0: [3, 0], 1: [1], 2: [0, 1], 3: [1, 2, 0], 4: [3], 5: [3, 1, 0]}
        for user, games in correct.items():
            self.assertEqual(games, recommendations[user])

class TestLOOCrossValidatoion(unittest.TestCase):
    def test_loo_popular(self):
        """Tests LOO recommendation algorithm on the popular recommendation
        algorithm by removing 75% of the reviews for each user and picking only
        the most popular recommendation."""
        def recommendation_algo(user, training_review_matrix):
            recommendations = recommend.popular_recommend([user], training_review_matrix, ranked_games)
            return recommendations[user][:1]

        alpha = 0.75 # remove all reviews for the user
        scores = score.loocv(alpha, review_matrix, recommendation_algo)
        correct = {0 : (0, 1, 1, 2), 1 : (1, 1, 0, 2), 2 : (1, 2, 0, 1),
                   3 : (0, 2, 1, 1), 4 : (0, 0, 1, 3), 5 : (0, 2, 1, 1)}
        self.assertEqual(scores, correct)

if __name__ == "__main__":
    unittest.main()
