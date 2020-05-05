import numpy as np
import numpy.ma as ma


'''
Predict rating for a certain user

Method: Weighted average

args:

returns: rating
'''


'''
Calculating the cosine similarity between two items

args: ratings1, ratings2 (array of ratings)

returns: similarity_score(a float)
'''
def CosineSimilarity(ratings1, ratings2):
    X = np.array(ratings1['ratings'])
    Y = np.array(ratings2['ratings'])
    return (np.dot(X, Y) / (np.sqrt(np.dot(X, X) * np.sqrt(np.dot(Y, Y)))))


'''
Calculating the similarity between two items

args: ratings1, ratings2 (arrays of ratings)
eg: ratings1 = [1, 9.0, 1.0, 10.0, 8.7]

returns: similarity_score (a float)
'''
def PearsonSimilarity(ratings1, ratings2):
    X = np.array(ratings1['ratings'])
    Y = np.array(ratings2['ratings'])
    return ma.corrcoef(ma.masked_invalid(X), ma.masked_invalid(Y))[0][1]

# Can be replaced with np.corrcoef(...)

'''
Creates a similarity matrix of the games
For the moment, we are using just the PearsonSimilarity

Optimizations: f(a, b) = f(b, a) so we might as well compute
               similarities for half of the matrix
               (However, are all the kernels gonna be symmetric??)

args: games, review_matrix, f (similarity_metric)

returns: similarity_matrix (as a np.array)
'''
def similarity_matrix(games_map, review_matrix, f=None):
    matrix = []
    games = games_map['forward']
    for key in games:
        for key2 in games:
            id1, id2 = games[key], games[key2]
            if (id1 == id2):
                matrix.append([1, id1, id2])
            else:
                data = data_arrange([id1, id2], review_matrix)
                matrix.append([PearsonSimilarity(data[0], data[1]), id1, id2])
    return np.array(matrix, ndmin=2)


'''
Finds which users rated the two games and 
returns two arrays (as a tuple) of those shared ratings
    
args: games_ids (array of ids of the two games), review_matrix
    
returns: (ratings1, ratings2)
'''
def data_arrange(games_ids, review_matrix):
    rating_1 = {'id': games_ids[0], 'ratings':[]}
    rating_2 = {'id': games_ids[1], 'ratings':[]}

    review_dict_1 = {}
    review_dict_2 = {}
    for elem in review_matrix.toarray():
        if (elem[2] == rating_1['id']):
            rating_1['ratings'].append(elem)
            review_dict_1[elem[1]] = elem
        elif (elem[2] == rating_2['id']):
            rating_2['ratings'].append(elem)
            review_dict_2[elem[1]] = elem
    
    '''Computing the intersection of the two rating arrays'''
    new_ratings_1 = []
    new_ratings_2 = []
    for elem in rating_2['ratings']:
        if (elem[1] in review_dict_1):
            new_ratings_2.append(elem[0])
    for elem in rating_1['ratings']:
        if (elem[1] in review_dict_2):
            new_ratings_1.append(elem[0])
    rating_1['ratings'] = new_ratings_1
    rating_2['ratings'] = new_ratings_2
    
    return (rating_1, rating_2)
