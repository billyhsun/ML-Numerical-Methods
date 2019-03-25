import numpy as np
import matplotlib.pyplot as plt
import util
from util import density_Gaussian
from itertools import product


def priorDistribution(beta):
    """
    Plot the contours of the prior distribution p(a)
    
    Inputs:
    ------
    beta: hyperparameter in the proir distribution
    
    Outputs: None
    -----
    """
    a = np.array([-0.1, -0.5])

    nx, ny = (100, 100)
    xticks = np.linspace(-1, 1, nx)
    yticks = np.linspace(-1, 1, ny)

    xv_2d, yv_2d = np.meshgrid(xticks, yticks, sparse=False)

    x_set = np.array(list(product(xticks, yticks)))

    mean_vec = np.array([0, 0])
    cov_mat = np.array([[beta, 0], [0, beta]])

    density = density_Gaussian(mean_vec, cov_mat, x_set)

    density = np.reshape(density, (nx, ny)).T

    contour = plt.contour(xv_2d, yv_2d, density)
    actual_point = plt.plot(a[0], a[1], 'ro', label='True value of a')

    plt.xlabel('a_0')
    plt.ylabel('a_1')
    plt.title('p(a)')
    plt.legend()

    plt.show()
    
    return 


def posteriorDistribution(x, z, beta, sigma2):
    """
    Plot the contours of the posterior distribution p(a|x,z)
    
    Inputs:
    ------
    x: inputs from training set
    z: targets from traninng set
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    
    Outputs: 
    -----
    mu: mean of the posterior distribution p(a|x,z)
    Cov: covariance of the posterior distribution p(a|x,z)
    """

    a = np.array([-0.1, -0.5])

    X_mat = np.column_stack((np.ones((x.shape[0], 1)), x))

    Cov = sigma2 * np.linalg.inv(np.matmul(X_mat.T, X_mat) + (sigma2/beta) * np.identity(x.shape[1]))

    mu = (1/sigma2) * np.matmul(Cov, np.matmul(X_mat.T, z))

    mu = (mu.T).squeeze()

    nx, ny = (100, 100)
    xticks = np.linspace(-1, 1, nx)
    yticks = np.linspace(-1, 1, ny)

    x_set = np.array(list(product(xticks, yticks)))
    xv_2d, yv_2d = np.meshgrid(xticks, yticks)

    density = density_Gaussian(mu, Cov, x_set)
    density = np.reshape(density, (nx, ny)).T

    contour = plt.contour(xv_2d, yv_2d, density, 10)
    actual_point = plt.plot(a[0], a[1], 'ro', label='True value of a')

    plt.xlabel('a_0')
    plt.ylabel('a_1')

    if x.shape[0] == 1:
        plt.title('p(a|x1,z1)')
    else:
        plt.title('p(a|x1,z1,..., x{},z{})'.format(x.shape[0], x.shape[0]))

    plt.legend()

    plt.show()

    return mu, Cov


def predictionDistribution(x, beta, sigma2, mu, Cov, x_train, z_train):
    """
    Make predictions for the inputs in x, and plot the predicted results 
    
    Inputs:
    ------
    x: new inputs
    beta: hyperparameter in the proir distribution
    sigma2: variance of Gaussian noise
    mu: output of posteriorDistribution()
    Cov: output of posteriorDistribution()
    x_train,z_train: training samples, used for scatter plot
    
    Outputs: None
    -----
    """
    x = np.array(x)

    '''
    Another way to compute the results and errors element-wise
    
    method2 = True

    if method2:
        mu_2 = np.array([mu[0]+data*mu[1] for data in x])
        var_2 = np.array([sigma2+ np.array([1,data]).T@Cov@np.array([1,data]) for data in x])
        st_dev2 = np.sqrt(var_2)

    '''

    X_mat = np.column_stack((np.ones((x.shape[0], 1)), x))

    print(Cov.shape)
    print(mu.shape)

    sigma2_out = sigma2 + np.matmul(np.matmul(X_mat, Cov),  X_mat.T)

    variances = np.diag(sigma2_out)
    st_devs = np.sqrt(variances)

    mu_out = X_mat @ mu

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Regression Result for {} training samples'.format(x_train.shape[0]))
    plt.errorbar(x, mu_out, yerr=st_devs, barsabove=True, ecolor='k', elinewidth=2, linewidth=3, capsize=3, color='r')
    plt.scatter(x_train, z_train, label='Training Samples', c='m')
    plt.legend()
    plt.show()

    return 


if __name__ == '__main__':
    
    # training data
    x_train, z_train = util.get_data_in_file('training.txt')
    # new inputs for prediction 
    x_test = [x for x in np.arange(-4.0, 4.01, 0.2)]
    
    # known parameters 
    sigma2 = 0.1
    beta = 5
    
    # number of training samples used to compute posterior
    ns = 5
    
    # used samples
    x = x_train[0:ns]
    z = z_train[0:ns]
    
    # prior distribution p(a)
    priorDistribution(beta)
    
    # posterior distribution p(a|x,z)
    mu, Cov = posteriorDistribution(x, z, beta, sigma2)
    
    # distribution of the prediction
    predictionDistribution(x_test, beta, sigma2, mu, Cov, x, z)
