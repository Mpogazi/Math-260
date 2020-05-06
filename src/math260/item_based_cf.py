import numpy as np
import numpy.ma as ma
from tqdm import tqdm


'''
Calculating the cosine similarity between two items

args: ratings1, ratings2 (array of ratings)

returns: similarity_score(a float)
'''
def CosineSimilarity(r1, r2):
    return (np.dot(r1, r2) / (np.sqrt(np.dot(r1, r1) * np.sqrt(np.dot(r2, r2)))))

'''
Calculating the similarity between two items

args: ratings1, ratings2 (arrays of ratings)
eg: ratings1 = [1, 9.0, 1.0, 10.0, 8.7]

returns: similarity_score (a float)
'''
def PearsonSimilarity(review_matrix, i, j):
    data = data_arrange(review_matrix, i, j)
    return np.corrcoef(data[0], data[1])[0][1]

# Can be replaced with np.corrcoef(...)

'''
Creates a similarity matrix of the games
For the moment, we are using just the PearsonSimilarity

Optimizations: f(a, b) = f(b, a) so we might as well compute
               similarities for half of the matrix
               (However, are all the kernels gonna be symmetric??)

args: games_map, review_matrix, f (similarity_metric)

returns: similarity_matrix (as a np.array)
'''
def sim_matrix(games_map, review_matrix, f):
    games = games_map['reverse']
    matrix = np.zeros((len(games), len(games)))
    for i in tqdm(range(len(games))):
        for j in range(len(games)):
            data = data_arrange(review_matrix, i, j)
            matrix[i, j] = f(data[0], data[1])
    return matrix

'''
Finds which users rated the two games and 
returns two arrays (as a tuple) of those shared ratings
    
args: review_matrix
    
returns: (ratings1, ratings2)
'''
def data_arrange(review_matrix, id1, id2):
    rating_1 = review_matrix[:,id1]
    rating_2 = review_matrix[:,id2]
    x_rmv = []
    y_rmv = []
    for i in range(len(rating_1)):
        x = rating_1[i]
        y = rating_2[i]
        if ((x == 0) or (y == 0)):
            x_rmv.append(i)
            y_rmv.append(i)
    
    return (np.delete(rating_1, x_rmv), np.delete(rating_2, y_rmv))