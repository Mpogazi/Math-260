import numpy as np
import random

def remove_reviews(alpha, user, review_matrix, randomized):
    """Removes a fraction alpha of the reviews from the specified user."""

    num_games = review_matrix.shape[1]
    reviewed_games = []
    for game in range(num_games):
        if review_matrix[user, game] != 0:
            reviewed_games.append(game)

    if randomized:
        random.shuffle(reviewed_games)

    num_to_remove = int(alpha * len(reviewed_games))
    for i in range(0, num_to_remove):
        game = reviewed_games[i]
        review_matrix[user, game] = 0

def loocv(alpha, review_matrix, recommendation_algorithm, randomized=False):
    """Performs LOO cross validation by removing a percent $\alpha$ of
    each users reviews and performing recommendation based upon that.
    
    Based upon a strategy from:

    http://www.bgu.ac.il/~shanigu/Publications/EvaluationMetrics.17.pdf

    Args:
       $\alpha$ : A fraction of user reviews to remove.
       review_matrix : A reviews matrix.
       recommendation algorithm : A function that takes a user and a 
       training set review_matrix and outputs a list of recommendations.

    Returns: 
       user_scores : A list where each index is a user ID and
       each element is a tuple (true_positives, true_negatives,
       false_positives, false_negatives) that is, the confusion
       matrix.
    """

    num_users = review_matrix.shape[0]
    user_scores = []
    for user in range(num_users):
        user_reviews = np.copy(review_matrix[user]) # save reviews

        # count positives + negatives
        N = sum(1 for _ in filter(lambda review: review == 0, user_reviews))
        P = len(user_reviews) - N

        remove_reviews(alpha, user, review_matrix, randomized) 
        recommendation = recommendation_algorithm(user, review_matrix) # perform recommendation
        review_matrix[user] = user_reviews # restore reviews

        # compute scores
        TP = sum(1 for _ in filter(lambda rec: user_reviews[rec] != 0, recommendation))
        FP = len(recommendation) - TP
        TN = N - FP
        FN = P - TP

        user_scores.append((TP, TN, FP, FN))
                                                                      
    return user_scores
