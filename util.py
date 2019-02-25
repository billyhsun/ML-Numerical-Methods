import os

def get_words_in_file(filename):
    """ Returns a list of all words in the file at filename. """
    with open(filename, 'r', encoding = "ISO-8859-1") as f:
        # read() reads in a string from a file pointer, and split() splits a
        # string into words based on whitespace
        words = f.read().split()
    return words

def get_files_in_folder(folder):
    """ Returns a list of files in folder (including the path to the file) """
    filenames = os.listdir(folder)
    # os.path.join combines paths while dealing with /s and \s appropriately
    full_filenames = [os.path.join(folder, filename) for filename in filenames]
    return full_filenames

def get_counts(file_list):
    """
    Returns a dict whose keys are words and whose values are the number of
    files in file_list the key occurred in.
    """
    counts = Counter()
    for f in file_list:
        words = get_words_in_file(f)
        for w in set(words):
            counts[w] += 1
    return counts

class Counter(dict):
    """
    Like a dict, but returns 0 if the key isn't found.
    """
    def __missing__(self, key):
        return 0


import numpy as np


def density_Gaussian(mean_vec, covariance_mat, x_set):
    """ Return the density of multivariate Gaussian distribution
        Inputs:
            mean_vec is a 1D array (like array([,,,]))
            covariance_mat is a 2D array (like array([[,],[,]]))
            x_set is a 2D array, each row is a sample
        Output:
            a 1D array, probability density evaluated at the samples in x_set.
    """
    d = x_set.shape[1]
    inv_Sigma = np.linalg.inv(covariance_mat)
    det_Sigma = np.linalg.det(covariance_mat)
    density = []
    for x in x_set:
        x_minus_mu = x - mean_vec
        exponent = - 0.5 * np.dot(np.dot(x_minus_mu, inv_Sigma), x_minus_mu.T)
        prob = 1 / (((2 * np.pi) ** (d / 2)) * np.sqrt(det_Sigma)) * np.exp(exponent)
        density.append(prob)
    density_array = np.array(density)

    return density_array


def get_data_in_file(filename):
    """
    Read the height/weight data and the labels from the given file as arrays
    """
    with open(filename, 'r') as f:
        data = []
        # read the data line by line
        for line in f:
            data.append([int(x) for x in line.split()])

            # store the height/weight data in x and the labels in y
    data_array = np.array(data)
    y = data_array[:, 0]  # labels
    x = data_array[:, 1:3]  # height/weight data

    return (x, y)
