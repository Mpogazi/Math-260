import numpy as np
import numpy.ma as ma
from tqdm import tqdm


'''
Calculating the cosine similarity between two items
args: a 2 column array
returns: similarity_score(a float)
'''
def CosineSimilarity(x):
    x = x[np.logical_not(np.logical_or(x[:,0] == 0, x[:,1] == 0))]
    return (np.dot(x[:,0], x[:,1]) / (np.sqrt(np.dot(x[:,0], x[:,0])) * np.sqrt(np.dot(x[:,1], x[:,1]))))

'''
Calculating the similarity between two items
args: a 2 column array
returns: similarity_score (a float)
'''
def PearsonSimilarity(x):
    x = x[np.logical_not(np.logical_or(x[:,0] == 0, x[:,1] == 0))]
    return np.corrcoef(x[:,0], x[:,1])[0, 1]

'''
Creates a similarity matrix
Caution: If the kernel is not symmetric, we have issues!
args: review_matrix, f (similarity_metric)
returns: similarity_matrix (as a np.array)
'''
def sim_matrix(review_matrix, bool_matrix, f):
    size = review_matrix.shape[1]
    matrix = np.zeros((size, size))
    # Completing the upper half of the function
    print('Building Similarity Matrices')
    for i in tqdm(range(size)):
        for j in range(i, size):
            matrix[i, j] = f(review_matrix[:,(i, j)], bool_matrix[:,(i,j)])
            matrix[j, i] = matrix[i, j]
    return matrix
