from math260 import data_prep, recommend, score
import matplotlib.pyplot as plt 
import numpy as np
from tqdm import tqdm

# choose your datasets
GAMES_DATA_FILE = "data/games.csv"
REVIEWS_FILE = "data/reviews.csv"

games, users = data_prep.parse_data(GAMES_DATA_FILE, REVIEWS_FILE, True)
games_map, users_map, rating_matrix, bool_matrix = \
    data_prep.create_review_matrix(games, users, sparse=False, verbose=True)

# creates a bar plot of the frequency of each rating

score_counts = np.zeros(11)

game_names = list(games.keys())

print('Iterating over all reviews')
for name in tqdm(game_names):
    game = games[name]
    reviews = game['reviews']
    for review in reviews:
        score_counts[int(review['rating'])] += 1

plt.bar(np.arange(11), score_counts)
plt.xlabel('Rating')
plt.ylabel('Number of Reviews')
plt.title('Distribution of Ratings')
plt.savefig('figures/rating-dist.png')

# creates a histogram of the average rating of each user

user_avg = np.sum(rating_matrix, axis=1) / np.sum(bool_matrix, axis=1)

plt.clf()
plt.hist(user_avg, bins=50)
plt.title('Distribution of Average User Rating')
plt.xlabel('Average Rating by User')
plt.ylabel('Number of Users')
plt.savefig('figures/user-avg-hist.png')

# creates a histogram of the average game rating

game_avg = np.sum(rating_matrix, axis=0) / np.sum(bool_matrix, axis= 0)

plt.clf()
plt.hist(game_avg, bins=50)
plt.title('Distribution of Average Game Rating')
plt.xlabel('Average Rating of Game')
plt.ylabel('Number of Games')
plt.savefig('figures/game-avg-hist.png')
