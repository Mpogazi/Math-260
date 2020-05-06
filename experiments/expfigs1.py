from math260 import data_prep, recommend, score
import matplotlib.pyplot as plt 
import numpy as np

# choose your datasets
GAMES_DATA_FILE = "data/2019-05-02-min.csv"
REVIEWS_FILE = "data/bgg-13m-reviews-min.csv"

games, users = data_prep.parse_data(GAMES_DATA_FILE, REVIEWS_FILE, True)

num_games = len(games)
num_users = len(users)

print('num games:\t\t{}'.format(num_games))
print('num users:\t\t{}'.format(num_users))

game_count = np.zeros(num_games)
user_count = np.zeros(num_users)

i = 0
for game in games:
    game_count[i] = len(games[game]['reviews'])
    i += 1
i = 0
for user in users:
    user_count[i] = len(users[user]['reviews'])
    i += 1

num_reviews = np.sum(game_count)

print('num reviews:\t\t{}'.format(num_reviews))

# puts the counts in descending order
game_count = np.sort(game_count)[::-1]
user_count = np.sort(user_count)[::-1]

# creates the histograms
plt.clf()
plt.hist(game_count, bins=200, log=True)
plt.title('Review Distribution over Games')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Games')
plt.savefig('figures/games-hist.png')

plt.clf()
plt.hist(user_count, bins=200, log=True)
plt.title('Review Distribution over Users')
plt.xlabel('Number of Reviews')
plt.ylabel('Number of Users')
plt.savefig('figures/users-hist.png')

game_cum = np.cumsum(game_count) / num_reviews
user_cum = np.cumsum(user_count) / num_reviews

# creates the cumulative plots
plt.clf()
plt.plot(np.arange(num_games)+1, game_cum)
plt.title('Distribution of Reviews Included as Number of Games Varies')
plt.xlabel('Number of Games Included')
plt.ylabel('Portion of Reviews Included')
plt.savefig('figures/games-cdf.png')

plt.clf()
plt.plot(np.arange(num_users)+1, user_cum)
plt.title('Distribution of Reviews Included as Number of Users Varies')
plt.xlabel('Number of Users Included')
plt.ylabel('Portion of Reviews Included')
plt.savefig('figures/users-cdf.png')