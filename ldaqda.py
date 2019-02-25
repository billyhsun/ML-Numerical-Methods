# Lab 1, Part 2: LDA and QDA Classifier

# Yuan Hong Sun

# 1003039838


import numpy as np
import matplotlib.pyplot as plt
import util
import math
import statistics


def LDA(height, weight, mu, sigma):
    sigma_inverse = np.linalg.inv(sigma)
    result = -0.5 * mu.T @ sigma_inverse @ mu + mu.T @ sigma_inverse @ np.array([height, weight]) + math.log(0.5)
    return result


def QDA(height, weight, mu, sigma):
    x = np.array([height, weight])
    sigma_inverse = np.linalg.inv(sigma)
    result = -0.5 * math.log(np.linalg.det(sigma)) - 0.5 * (x - mu).T @ sigma_inverse @ (x - mu) + math.log(0.5)
    return result


def discrimAnalysis(x, y):
    """
    Estimate the parameters in LDA/QDA and visualize the LDA/QDA models
    
    Inputs
    ------
    x: a N-by-2 2D array contains the height/weight data of the N samples
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of five elments: mu_male,mu_female,cov,cov_male,cov_female
    in which mu_male, mu_female are mean vectors (as 1D arrays)
             cov, cov_male, cov_female are covariance matrices (as 2D arrays)
    Besides producing the five outputs, you need also to plot 1 figure for LDA 
    and 1 figure for QDA in this function         
    """
    ### TODO: Write your code here

    ''' Calculating Covariance '''

    N = y.shape[0]

    # Calculate means
    x_male = []
    x_female = []
    for i in range(N):
        if y[i] == 1:
            x_male.append(x[i])
        else:
            x_female.append(x[i])

    x_male = np.array(x_male)
    x_female = np.array(x_female)

    mean_male = np.mean(x_male, axis=0)
    mean_female = np.mean(x_female, axis=0)

    # Calculate covariance matrices for male and female
    covariance_male = np.zeros([2, 2])
    covariance_female = np.zeros([2, 2])

    for male in x_male:
        covariance_male += np.outer(male - mean_male, male - mean_male)

    for female in x_female:
        covariance_female += np.outer(female - mean_female, female - mean_female)

    covariance_male = covariance_male / len(x_male)
    covariance_female = covariance_female / len(x_female)

    # Calculate overall covariance matrix
    covariance = np.zeros([2, 2])
    mean = np.mean(x, axis=0)
    for person in x:
        covariance += np.outer(person - mean, person - mean)
    covariance = covariance / N

    ''' Plots '''

    # Initialize Plots
    nx, ny = (100, 100)
    x_ticks = np.linspace(50, 80, nx)
    y_ticks = np.linspace(80, 280, ny)
    xv, yv = np.meshgrid(x_ticks, y_ticks, sparse=True)
    xv_2d, yv_2d = np.meshgrid(x_ticks, y_ticks, sparse=False)

    heights = np.linspace(50, 80, 100)
    weights = np.linspace(80, 280, 100)
    heightgrid, weightgrid = np.meshgrid(heights, weights)

    h = heightgrid.ravel()
    w = weightgrid.ravel()

    # LDA Plot

    LDA_density_male = []
    LDA_density_female = []

    for i in range(100):
        temp_male = []
        temp_female = []
        for j in range(100):
            temp_male.append(LDA(xv_2d[j][i], yv_2d[j][i], mean_male, covariance))
            temp_female.append(LDA(xv_2d[j][i], yv_2d[j][i], mean_female, covariance))
        LDA_density_male.append(np.array(temp_male))
        LDA_density_female.append(np.array(temp_female))

    LDA_density_male = np.array(LDA_density_male)
    LDA_density_female = np.array(LDA_density_female)
    LDA_boundary = (LDA_density_male - LDA_density_female).T

    # Plot contours
    heightweight = np.concatenate((h.reshape(-1, 1), w.reshape(-1, 1)), axis=1)
    LDAResM = util.density_Gaussian(mean_male, covariance, heightweight)
    LDAResM = LDAResM.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid, weightgrid, LDAResM)
    # plt.clabel(ctm, fontsize = 8, fmt = r'LDA Male Contours')

    LDAResF = util.density_Gaussian(mean_female, covariance, heightweight)
    LDAResF = LDAResF.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid, weightgrid, LDAResF)
    # plt.clabel(ctm, fontsize = 8, fmt = r'LDA Female Contours')

    # male_contours = plt.contour(xv_2d, yv_2d, LDA_density_male, 5, linestyles='dashed', colors='blue')
    # female_contours = plt.contour(xv_2d, yv_2d, LDA_density_female, 5, colors='red')
    # plt.clabel(male_contours)
    # plt.clabel(female_contours)

    boundary = plt.contour(xv_2d, yv_2d, LDA_boundary, [0])
    plt.clabel(boundary, fontsize=10, inline=2, fmt=r'LDA Boundary')

    males = plt.scatter(*zip(*x_male), marker='o', color='blue')
    females = plt.scatter(*zip(*x_female), marker='o', color='red')

    plt.xlim(50, 80)
    plt.ylim(80, 285)
    plt.legend((males, females), ('Males', 'Females'), loc='upper left')
    plt.xlabel('Height')
    plt.ylabel('Weight')

    plt.show()
    plt.clf()
    plt.close()

    # QDA Plot

    QDA_density_male = []
    QDA_density_female = []

    for i in range(100):
        temp_male = []
        temp_female = []
        for j in range(100):
            temp_male.append(QDA(xv_2d[j][i], yv_2d[j][i], mean_male, covariance_male))
            temp_female.append(QDA(xv_2d[j][i], yv_2d[j][i], mean_female, covariance_female))
        QDA_density_male.append(np.array(temp_male))
        QDA_density_female.append(np.array(temp_female))

    QDA_density_male = np.array(QDA_density_male)
    QDA_density_female = np.array(QDA_density_female)
    QDA_boundary = (QDA_density_male - QDA_density_female).T

    # QDA_male_contours = plt.contour(xv_2d, yv_2d, QDA_male_density, 5, linestyles='dashed', colors='blue')
    # QDA_female_contours = plt.contour(xv_2d, yv_2d, QDA_female_density, 5, colors='red')
    QDA_boundary = plt.contour(xv_2d, yv_2d, QDA_boundary, [0])

    # Plot contours
    QDAResM = util.density_Gaussian(mean_male, covariance_male, heightweight)
    QDAResM = QDAResM.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid, weightgrid, QDAResM)
    # plt.clabel(ctm, fontsize = 8, fmt = r'LDA Male Contours')

    QDAResF = util.density_Gaussian(mean_female, covariance_female, heightweight)
    QDAResF = QDAResF.reshape(heightgrid.shape)
    ctm = plt.contour(heightgrid, weightgrid, QDAResF)
    # plt.clabel(ctm, fontsize = 8, fmt = r'LDA Female Contours')

    # plt.clabel(QDA_female_contours)
    plt.clabel(QDA_boundary, fontsize=10, inline=2, fmt=r'QDA Boundary')

    males = plt.scatter(*zip(*x_male), marker='o', color='blue')
    females = plt.scatter(*zip(*x_female), marker='o', color='red')

    plt.xlim(50, 80)
    plt.ylim(80, 285)
    plt.legend((males, females), ('Males', 'Females'), loc='upper left')
    plt.xlabel('Height')
    plt.ylabel('Weight')

    plt.show()
    plt.clf()
    plt.close()

    return mean_male, mean_female, covariance, covariance_male, covariance_female
    

def misRate(mu_male,mu_female,cov,cov_male,cov_female,x,y):
    """
    Use LDA/QDA on the testing set and compute the misclassification rate
    
    Inputs
    ------
    mu_male,mu_female,cov,cov_male,mu_female: parameters from discrimAnalysis
    
    x: a N-by-2 2D array contains the height/weight data of the N samples  
    
    y: a N-by-1 1D array contains the labels of the N samples 
    
    Outputs
    -----
    A tuple of two elements: (mis rate in LDA, mis rate in QDA )
    """
    ### TODO: Write your code here
   
    result_LDA = []
    result_QDA = []

    for person in x:
        if LDA(person[0], person[1], mu_male, cov) > LDA(person[0], person[1], mu_female, cov):
            result_LDA.append(1)
        else:
            result_LDA.append(2)

        if QDA(person[0], person[1], mu_male, cov_male) > QDA(person[0], person[1], mu_female, cov_female):
            result_QDA.append(1)
        else:
            result_QDA.append(2)

    mis_lda = (np.array(result_LDA) != y).sum() / y.shape[0]
    mis_qda = (np.array(result_QDA) != y).sum() / y.shape[0]

    return (mis_lda, mis_qda)


if __name__ == '__main__':
    
    # load training data and testing data
    x_train, y_train = util.get_data_in_file('trainHeightWeight.txt')
    x_test, y_test = util.get_data_in_file('testHeightWeight.txt')
    
    # parameter estimation and visualization in LDA/QDA
    mu_male,mu_female,cov,cov_male,cov_female = discrimAnalysis(x_train,y_train)
    
    # misclassification rate computation
    mis_LDA,mis_QDA = misRate(mu_male,mu_female,cov,cov_male,cov_female,x_test,y_test)

    print("The misclassifcation rate for LDA is {}, for QDA is: {}".format(mis_LDA.round(3), mis_QDA.round(3)))
