import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import random

data = pd.read_csv("/Users/fanyang/Dropbox/uiuc/cs598/quiz/quiz4.csv")
# 658763549

# E-Step
def Estep(data, G, para):

    n = data.shape[0]
    result = np.ones((n, G))

    P = para['Prob']
    Mu = para['Mean']
    Sigma = para['Variance']

    p_1 = P[0]
    mu_1 = Mu[:, 0].reshape(-1,1)
    Inv_Sigma = np.linalg.inv(Sigma)

    for i in range(n):
        # i = 1
        x_i = np.array(data.iloc[i]).reshape(-1,1)
        x1_mu = x_i - mu_1
        denom = 0
        for g in range(G):
            #g = 1
            mu_g = Mu[:, g].reshape(-1,1)
            p_g = P[g]
            # a_g = np.log(p_g * multivariate_normal.pdf(x_i, mean=mu_g, cov=Sigma)) - np.log(
            #     p_1 * multivariate_normal.pdf(x_i, mean=mu_1, cov=Sigma))
            xi_mu = x_i - mu_g
            a_g = np.log(p_g/p_1) +  0.5*x1_mu.T.dot(Inv_Sigma).dot(x1_mu) - 0.5*xi_mu.T.dot(Inv_Sigma).dot(xi_mu)
            num_g = np.exp(a_g)
            denom += num_g


        for g in range(G):
            mu_g = Mu[:,g].reshape(-1,1)
            p_g = P[g]
            xi_mu = x_i - mu_g
            a_g = np.log(p_g / p_1) + 0.5 * x1_mu.T.dot(Inv_Sigma).dot(x1_mu) - 0.5 * xi_mu.T.dot(Inv_Sigma).dot(xi_mu)
            num_g = np.exp(a_g)
            b_g = num_g/denom

            result[i,g] = b_g
            # print(b_g)

    return result


# Mstep
# post_prob is the result from Estep
def Mstep(data, G, para, post_prob):

    # P
    P = post_prob.mean(axis = 0)
    para['Prob'] = P
    # Mu
    Mu = para['Mean']

    for g in range(G):
        den = post_prob[:,g].sum()
        num = post_prob[:,g].reshape(1,-1).dot(data)
        updated_mu_g = num/den
        Mu[:,g] = updated_mu_g
    # para['Mean'] = Mu
    # Sigma
    n = data.shape[0]

    dim = data.shape[1]
    updated_Sigma = np.ones((dim,dim))*0.0

    for g in range(G):
        prob_g = post_prob[:,g]
        data_mu = data-Mu[:,g]
        data_prob = data_mu*prob_g[:,np.newaxis]
        updated_Sigma += data_mu.T.dot(data_prob)

    para['Variance'] = updated_Sigma/n
    return para

# for i in range(n):
#     for g in range(G):
#         r_ig = post_prob[i, g]
#         x_ig = np.array(data.iloc[i] - Mu[:,g]).reshape(-1,1)
#         s_ig = r_ig*x_ig.dot(x_ig.T)
#         updated_Sigma += s_ig




def MyEM(data, itmax, G, para):
    round = 1
    current_para = para
    while round<=itmax:
        print(round)
        print(current_para)
        post_prob = Estep(data, G, current_para)
        current_para = Mstep(data, G, current_para, post_prob)
        round += 1
    return para




#
n_G = 2
# P = np.ones((n_G,1))
P = np.array([0.4926471, 0.5073529])
# P[1,0] = 0.4816176
# P[0,0] = 0.5183824
Mu = np.ones((data.shape[1], n_G))
# [index1, index2] = random.sample(range(data.shape[0]), 2)
Mu[:, 0] = np.array([3.411925,70.380597])
Mu[:, 1] = np.array([3.561442,71.398551])
# data.iloc[index2].to_numpy()

# Sigma = data.cov()*0.6
# Sigma = np.ones((2,2))
Sigma = np.array([[1.292351, 13.88838], [13.888377,183.88481]])
para = {'Prob': P, 'Mean': Mu, 'Variance': Sigma}

final_para = MyEM(data, 21, n_G, para)

# dim = data.shape[1]
# updated_Sigma = np.ones((dim, dim)) * 0.0
# for i in range(n):
#     for g in range(G):



#
n_G = 3
P = np.array([0.3235294, 0.3455882, 0.3308824])
Mu = np.ones((data.shape[1], n_G))
Mu[:, 0] = np.array([3.431989,70.090909])
Mu[:, 1] = np.array([3.487043,71.053191])
Mu[:, 2] = np.array([3.543111,71.522222])
Sigma = np.array([[1.295919,13.90046], [13.900462, 183.79582]])
para = {'Prob': P, 'Mean': Mu, 'Variance': Sigma}
final_para = MyEM(data, 21, n_G, para)


# from sample
# n_G = 2
#
# P = np.array([0.5, 0.5])
#
# Mu = np.ones((data.shape[1], n_G))
# Mu[:, 0] = np.array([3.467750,70.132353])
# Mu[:, 1] = np.array([3.5078162,71.6617647])
#
# Sigma = np.array([[1.2975376, 13.911099], [13.9110994,183.559040]])
#
# para = {'Prob': P, 'Mean': Mu, 'Variance': Sigma}
# final_para = MyEM(data, 20, n_G, para)
# final_para
# $prob
# [1] 0.50062804 0.49937196
#
# $mean
#                [,1]       [,2]
# eruptions  3.467639  3.5079778
# waiting   70.118091 71.6779860
#
# $Sigma
#            eruptions    waiting
# eruptions  1.2975321  13.910688
# waiting   13.9106878 183.535498
#
# n_G = 3
# P = np.array([0.30514706,0.34926471,0.34558824])
# Mu = np.ones((data.shape[1], n_G))
# Mu[:, 0] = np.array([3.4459639,69.8433735])
# Mu[:, 1] = np.array([3.6217053,72.1578947])
# Mu[:, 2] = np.array([3.3893617, 70.5531915])
# Sigma = np.array([[1.2877935,13.842302], [13.8423020,183.208932]])
# para = {'Prob': P, 'Mean': Mu, 'Variance': Sigma}
# final_para = MyEM(data, 20, n_G, para)
# final_para
# $prob
# [1] 0.30512459 0.34979670 0.34507871
#
# $mean
#                 [,1]       [,2]       [,3]
# eruptions  3.4486293  3.6045405  3.4040498
# waiting   69.9091848 71.8187707 70.8362404
#
# $Sigma
#            eruptions    waiting
# eruptions  1.2902832  13.875216
# waiting   13.8752157 183.547598