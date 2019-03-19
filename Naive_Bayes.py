# Lab 1, Part 1: Naive Bayesian Classifier

# Yuan Hong Sun

# 1003039838


import os.path
import numpy as np
import matplotlib.pyplot as plt
import util
import itertools
import math


def learn_distributions(file_lists_by_category):
    """
    Estimate the parameters p_d, and q_d from the training set
    
    Input
    -----
    file_lists_by_category: A two-element list. The first element is a list of 
    spam files, and the second element is a list of ham files.

    Output
    ------
    probabilities_by_category: A two-element tuple. The first element is a dict 
    whose keys are words, and whose values are the smoothed estimates of p_d;
    the second element is a dict whose keys are words, and whose values are the 
    smoothed estimates of q_d 
    """
    ### TODO: Write your code here

    spam_emails = file_lists_by_category[0]
    ham_emails = file_lists_by_category[1]

    spam_email_word_counts = util.get_counts(spam_emails)
    ham_email_word_counts = util.get_counts(ham_emails)

    file_list_train = list(itertools.chain.from_iterable(file_lists_by_category))
    N = len(file_list_train)
    vocabulary = set(util.get_counts(file_list_train).keys())
    D = len(vocabulary)

    words_p_d = {}
    words_q_d = {}

    for word in vocabulary:
        words_p_d[word] = (spam_email_word_counts[word] + 1) / (len(spam_emails) + 2)
        words_q_d[word] = (ham_email_word_counts[word] + 1) / (len(ham_emails) + 2)

    probabilities_by_category = (words_p_d, words_q_d)

    return probabilities_by_category


def classify_new_email(filename, probabilities_by_category, prior_by_category, adjustment):
    """
    Use Naive Bayes classification to classify the email in the given file.

    Inputs
    ------
    filename: name of the file to be classified
    
    probabilities_by_category: output of function learn_distributions
    
    prior_by_category: A two-element list as [\pi, 1-\pi], where \pi is the 
    parameter in the prior class distribution

    Output
    ------
    classify_result: A two-element tuple. The first element is a string whose value
    is either 'spam' or 'ham' depending on the classification result, and the 
    second element is a two-element list as [log p(y=1|x), log p(y=0|x)], 
    representing the log posterior probabilities
    """
    ### TODO: Write your code here

    D = len(probabilities_by_category)

    # Prior distributions (initital)
    MAP_spam = math.log(prior_by_category[0])
    MAP_ham = math.log(prior_by_category[1])

    all_words = util.get_words_in_file(filename)
    vocab = list(probabilities_by_category[0].keys())

    # Calculate for each subsequent word
    for word in vocab:
        if word in all_words:
            MAP_spam += math.log(probabilities_by_category[0][word])
            MAP_ham += math.log(probabilities_by_category[1][word])

        else:
            MAP_spam += math.log(1-probabilities_by_category[0][word])
            MAP_ham += math.log(1-probabilities_by_category[1][word])

    # Check for the result
    if MAP_spam > adjustment * MAP_ham:
        result = 'spam'
    else:
        result = 'ham'

    classify_result = (result, [MAP_spam, MAP_ham])

    return classify_result


if __name__ == '__main__':
    
    # folder for training and testing 
    spam_folder = "data/spam"
    ham_folder = "data/ham"
    test_folder = "data/testing"

    # generate the file lists for training
    file_lists = []
    for folder in (spam_folder, ham_folder):
        file_lists.append(util.get_files_in_folder(folder))
        
    # Learn the distributions    
    probabilities_by_category = learn_distributions(file_lists)
    
    # prior class distribution
    priors_by_category = [0.5, 0.5]
    
    # Store the classification results
    performance_measures = np.zeros([2,2])
    # explanation of performance_measures:
    # columns and rows are indexed by 0 = 'spam' and 1 = 'ham'
    # rows correspond to true label, columns correspond to guessed label
    # to be more clear, performance_measures = [[p1 p2]
    #                                           [p3 p4]]
    # p1 = Number of emails whose true label is 'spam' and classified as 'spam' 
    # p2 = Number of emails whose true label is 'spam' and classified as 'ham' 
    # p3 = Number of emails whose true label is 'ham' and classified as 'spam' 
    # p4 = Number of emails whose true label is 'ham' and classified as 'ham' 

    ### TODO: Write your code here to modify the decision rule such that
    ### Type 1 and Type 2 errors can be traded off, plot the trade-off curve

    # Classify emails from testing set and measure the performance
    type1_error_list = []
    type2_error_list = []

    # Use different adjustments
    for i in np.linspace(0.8, 1.2, 100):
        performance_measures = np.zeros([2, 2])
        for filename in (util.get_files_in_folder(test_folder)):
            # Classify
            label, log_posterior = classify_new_email(filename,
                                                      probabilities_by_category,
                                                      priors_by_category, i)

            # Measure performance (the filename indicates the true label)
            base = os.path.basename(filename)
            true_index = ('ham' in base)
            guessed_index = (label == 'ham')
            performance_measures[int(true_index), int(guessed_index)] += 1

        template = "You correctly classified %d out of %d spam emails, and %d out of %d ham emails."
        # Correct counts are on the diagonal
        correct = np.diag(performance_measures)
        # totals are obtained by summing across guessed labels
        totals = np.sum(performance_measures, 1)
        print(template % (correct[0], totals[0], correct[1], totals[1]))

        type1_error = totals[0] - correct[0]
        type2_error = totals[1] - correct[1]
        type1_error_list.append(type1_error)
        type2_error_list.append(type2_error)

    plt.plot(type1_error_list, type2_error_list, marker='x', markersize=8, markerfacecolor='red', color='black', linestyle='-', linewidth='2.0')

    plt.xlabel('Number of Type 1 Errors')
    plt.ylabel('Number of Type 2 Errors')
    plt.title('Tradeoff Curve')
    plt.show()

   

 