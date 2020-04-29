import csv
import scipy.sparse as sp

def minimize_games(games_path, output_path, verbose=False):
    """
    Reads in the games and strips out the ID, Year, URL, and thumbnail,
    as well as renames the headers to use snake_case
    -------------------------
    games_path - filepath to the game set
    output_path - filepath to write the minimized version
    verbose - Whether to log progress
    """
    if verbose:
        print('Minimizing games')
        print(' - Opening file')
    games_file = open(games_path)
    games_reader = csv.DictReader(games_file)

    if verbose:
        print(' - Creating output file')

    output_file = open(output_path, mode='w')
    fields = ['name', 'rank', 'average', 'bayes_average', 'users_rated']
    writer = csv.DictWriter(output_file, fieldnames=fields)
    writer.writeheader()

    if verbose:
        print(' - Reading and minimizing')

    for game in games_reader:
        g = {
            'name': game['Name'],
            'rank': game['Rank'],
            'average': game['Average'],
            'bayes_average': game['Bayes average'],
            'users_rated': game['Users rated']
        }
        writer.writerow(g)
    
    if verbose:
        print(' - Closing files')
    
    games_file.close()
    output_file.close()

    if verbose:
        print(' - Minimization complete\n')


def minimize_reviews(reviews_path, output_path, verbose=False):
    """
    Reads in the reviews and strips out the comments, ID, and index fields
    then saves the smaller file to the given location
    -------------------------
    reviews_path - filepath to the review set
    output_path - filepath to write the minimized version
    verbose - Whether to log progress
    """
    if verbose:
        print('Minizing reviews')
        print(' - Opening file')
    reviews_file = open(reviews_path)
    reviews_reader = csv.DictReader(reviews_file)
    
    if verbose:
        print(' - Creating output file')
    
    output_file = open(output_path, mode='w')
    fields = ['user', 'rating', 'name']
    writer = csv.DictWriter(output_file, fieldnames=fields)
    writer.writeheader()

    if verbose:
        print(' - Reading and minimizing (~13.2m reviews)')
    
    i = 0
    for review in reviews_reader:
        if i % 1000000 == 0 and verbose:
            print('   - '+str(i)+' reviews minimized')
        i += 1
        min_review = {
            'user': review['user'],
            'rating': review['rating'],
            'name': review['name']
        }
        writer.writerow(min_review)

    if verbose:
        print(' - Closing files')

    reviews_file.close()
    output_file.close()

    if verbose:
        print(' - Minimization complete\n')
    
def build_debug(o_games_path, d_games_path, o_reviews_path, d_reviews_path,
                max_games, max_reviews, verbose=False):
    """
    Reads in the files for games and (minimzed) reviews and generates a new
    dataset which has at most max_games and max_reviews in it.
    -------------------------
    o_games_path - filepath to the games dataset
    d_games_path - filepath to write the debug games dataset
    o_review_path - filepath to the (minimized) review dataset 
    d_review_path - filepath to write the debug reviews dataset
    max_games - maximum number of games to include
    max_reviews - maximum number of reviews to include
    verbose - flag of whether to log progress
    """
    if verbose:
        print('Building Debug')
        print(' - Opening files')
    og_file = open(o_games_path)
    or_file = open(o_reviews_path)

    games_dict = {} # dict to track which games we consider
    reviews = [] # list of reviews to track
    games_count = 0 # number of games included so far
    review_count = 0 # number of reviews included so far

    if verbose:
        print(' - Reading reviews')
    or_reader = csv.DictReader(or_file)

    for review in or_reader:
        game_name = review['name']
        # check if we have seen the game yet
        if not game_name in games_dict:
            # check if we are accepting new games
            if games_count == max_games:
                # not accepting anymore so discard the review
                continue
            else:
                # still accepting so add the game to the dict and count it
                games_dict[game_name] = True
                games_count += 1
        reviews.append(review)
        review_count += 1
        if review_count == max_reviews:
            break
    or_file.close()
    
    if verbose:
        print(' - Reviews have been selected')
        print(' - Picking out the necessary games')
    og_reader = csv.DictReader(og_file)

    games = [] # list of games to track

    for game in og_reader:
        if not game['name'] in games_dict:
            # we ignore the games not in the dict
            continue
        games.append(game)
    og_file.close()

    if verbose:
        print(' - All games picked out')
        print(' - Writing review debug dataset')

    dr_file = open(d_reviews_path, mode='w')
    r_fields = ['user', 'rating', 'name']
    r_writer = csv.DictWriter(dr_file, fieldnames=r_fields)
    r_writer.writeheader()

    for review in reviews:
        r_writer.writerow(review)
    dr_file.close()

    if verbose:
        print(' - Writing game debug dataset')

    dg_file = open(d_games_path, mode='w')
    g_fields = ['name', 'rank', 'average', 'bayes_average', 'users_rated']
    g_writer= csv.DictWriter(dg_file, fieldnames=g_fields)
    g_writer.writeheader()

    for game in games:
        g_writer.writerow(game)
    dg_file.close()
    
    if verbose:
        print(' - Debug datasets complete\n')

def parse_data(games_path, reviews_path, verbose=False):
    """
    Reads in the files of game data and reviews and spits out dictionaries
    correlating games with users based on the reviews.
    -------------------------
    games_path - filepath to the basic game data
    reviews_path - filepath to the review set
    verbose - flag of whether to log progress
    """
    if verbose:
        print('Parsing data')
        print(' - Opening files')
    gd_file = open(games_path)
    reviews_file = open(reviews_path)

    if verbose:
        print(' - Reading files')
        print('   - Reading games data')
    gd_reader = csv.DictReader(gd_file)
    games = {}
    for game in gd_reader:
        # parsing out some fields
        game['Rank'] = int(game['rank'])
        game['average'] = float(game['average'])
        game['bayes_average'] = float(game['bayes_average'])
        game['users_rated'] = int(game['users_rated'])
        # adding these in for later
        game['reviews'] = [] 
        game['users'] = []
        game['users_dict'] = {}

        games[game['name']] = game
    gd_file.close()

    if verbose:
        print('   - Reading reviews and correlating')
    reviews_reader = csv.DictReader(reviews_file)
    users = {}
    i = 0
    for review in reviews_reader:
        if verbose and i % 100000 == 0:
            print('     - '+str(i))
        i += 1
        user_name = review['user']
        game_name = review['name']
        review['rating'] = float(review['rating'])
        game = games[game_name]

        # associating review and game to user
        if user_name in users:
            user = users[user_name]
            user['reviews'].append(review)
            user['games'].append(game)
            user['games_dict'][game_name] = game
        else:
            user = {
                'name': user_name,
                'reviews': [review],
                'games': [game],
                'games_dict': {
                    game_name: game
                }
            }
            users[user_name] = user
        
        # associateing review and user to game
        game['reviews'].append(review)
        game['users'].append(user)
        game['users_dict']['user_name'] = user        
    reviews_file.close()

    if verbose:
        print(' - Parsing data complete\n')
    return games, users

def create_review_matrix(games, users, verbose=False):
    '''
    Using the games and users dataset creates a sparse matrix of reviews,
    a sparse matrix of if a user has done a review, and maps from 
    games/users to matrix index
    -------------------------
    games - the games dataset
    users - the users dataset
    verbose - whether to log progress
    '''

    if verbose:
        print('Creating review matrix')

    games_map = {
        'forward': {},
        'reverse': []
    }
    users_map = {
        'forward': {},
        'reverse': []
    }

    if verbose:
        print(' - Creating games map')

    index = 0
    for game_name in games:
        games_map['forward'][game_name] = index
        games_map['reverse'].append(game_name)
        index += 1

    if verbose:
        print(' - Creating users map')
    
    index = 0
    for user_name in users:
        users_map['forward'][user_name] = index
        users_map['reverse'].append(user_name)
        index += 1

    if verbose:
        print(' - Generating coordinates')

    rating_coords = []
    bool_coords = []
    for game_name in games:
        game_index = games_map['forward'][game_name]
        game = games[game_name]
        for review in game['reviews']:
            user_name = review['user']
            rating = review['rating']
            user_index = users_map['forward'][user_name]
            rating_coords.append([rating, user_index, game_index])
            bool_coords.append([1, user_index, game_index])
    
    if verbose:
        print(' - Creating sparse matrices')

    rating_matrix = sp.coo_matrix(rating_coords)
    bool_matrix = sp.coo_matrix(bool_coords)

    if verbose:
        print(' - Matrices created\n')

    return games_map, users_map, rating_matrix, bool_matrix
    

if __name__ == "__main__":
    original_games = "data/2019-05-02.csv"
    original_reviews = "data/bgg-13m-reviews.csv"
    minimized_games = "data/2019-05-02-min.csv"
    minimized_reviews = "data/bgg-13m-reviews-min.csv"
    debug_games = "data/2019-05-02-debug.csv"
    debug_reviews = "data/bgg-13m-reviews-debug.csv"
    minimize_games(original_games, minimized_games, True)
    minimize_reviews(original_reviews, minimized_reviews, True)
    build_debug(minimized_games, debug_games, minimized_reviews, debug_reviews,
                 1000, 1000000, True)