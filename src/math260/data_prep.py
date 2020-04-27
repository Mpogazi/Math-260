import csv

def minimize_reviews(reviews_path, output_path, verbose=False):
    """
    Reads in the reviews and strips out the comments, ID, and index fields
    then saves the smaller file to the given location
    -------------------------
    reviews_path - filepath to the review set
    output_path - filepath to write the minimized version
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
        game['Rank'] = int(game['Rank'])
        game['Average'] = float(game['Average'])
        game['Bayes average'] = float(game['Bayes average'])
        game['Users rated'] = int(game['Users rated'])
        # adding these in for later
        game['reviews'] = [] 
        game['users'] = []
        game['users_dict'] = {}

        games[game['Name']] = game
    gd_file.close()

    #if True:
    #    return games, None
    '''
    if verbose:
        print('   - Reading games detailed data')
    gdd_reader = csv.DictReader(gdd_file)
    for details in gdd_reader:
        if details['primary'] == 'Wooly Wars':
            print(details)
        game = games[details['primary']]
        game['details'] = details
    gdd_file.close()
    '''

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
        print(' - Parsing data complete')
    return games, users

if __name__ == "__main__":
    original_reviews = "data/bgg-13m-reviews.csv"
    minimized_reviews = "data/bgg-13m-reviews-min.csv"
    minimize_reviews(original_reviews, minimize_reviews, True)