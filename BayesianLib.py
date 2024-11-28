import numpy as np
import matplotlib.pyplot as plt
from ERADist import ERADist
from ERANataf import ERANataf
from aBUS_SuS import aBUS_SuS
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from scipy.stats import spearmanr
from scipy.spatial import distance

# ------------------------ Bayesian Linear Regression ---------------------------------------------------------------

class BayesianRegression():

    def __init__(self, function, numberParam, correlated_errors=False):
        self.function = function
        self.numberParam = numberParam
        self.correlated_errors = correlated_errors
        return

    def addData(self, x, y):
        self.x = x #imposed stresses
        self.y = y #measured creep

    def setPrior(self, prior_pdf_input):

        prior_pdf = []
        for i in range(len(prior_pdf_input)):
            if self.correlated_errors == False:
                if i == len(prior_pdf_input)-1:
                    prior_pdf_i = ERADist(prior_pdf_input[i][0], 'PAR', [prior_pdf_input[i][1], prior_pdf_input[i][2]])
                else:
                    prior_pdf_i = ERADist(prior_pdf_input[i][0], 'MOM', [prior_pdf_input[i][1], prior_pdf_input[i][2]])
            elif self.correlated_errors == True:
                if i == len(prior_pdf_input)-3 or i == len(prior_pdf_input)-2 or i == len(prior_pdf_input)-1:
                    prior_pdf_i = ERADist(prior_pdf_input[i][0], 'PAR', [prior_pdf_input[i][1], prior_pdf_input[i][2]])
                else:
                    prior_pdf_i = ERADist(prior_pdf_input[i][0], 'MOM', [prior_pdf_input[i][1], prior_pdf_input[i][2]])

            prior_pdf.append(prior_pdf_i)


        
        d = len(prior_pdf)
        R = np.eye(d) # Independent case

        self.prior = ERANataf(prior_pdf, R)
        # TODO Make the dependent case as well here.

    def cov_matrix(self, param, x):
        x_dist = np.resize(x, (1,len(x)))
        dist_matrix = distance.cdist(x_dist.transpose(), x_dist.transpose(), 'euclidean')

        cov_matrix = np.exp((-2 * abs(dist_matrix)) / param)

        return cov_matrix

    def Likelihood(self):
        if self.correlated_errors == False:
            f = lambda param: self.function(self.x, param)
            realmin = np.finfo(np.double).tiny  # to prevent log(0)
            likelihood = lambda param: mvn.pdf(f(param), self.y, np.diag(param[-1] ** 2 * np.ones(len(self.y))))
            self.log_likelihood = lambda param: np.log(likelihood(param) + realmin)
        elif self.correlated_errors == True:
            f = lambda param: self.function(self.x, param)
            realmin = np.finfo(np.double).tiny  # to prevent log(0)

            self.var_measurement = lambda param, x: (np.diag(param ** 2 * np.ones(len(x))))
            self.var_model = lambda param, x: self.cov_matrix(param, x)

            likelihood = lambda param: mvn.pdf(f(param), self.y.flatten(), (self.var_measurement(param[-3], self.x)+param[-1]*self.var_model(param[-2], self.x)))
            self.log_likelihood = lambda param: np.log(likelihood(param) + realmin)


    def regression(self, prior_pdf, N = int(10e3), p0 = 0.1):
        self.N = N
        self.setPrior(prior_pdf)
        self.Likelihood()

        print('\naBUS with SUBSET SIMULATION: \n')
        h, samplesU, samplesX, logcE, c, sigma = aBUS_SuS(N, p0, self.log_likelihood, self.prior)

        nsub = len(h.flatten()) + 1  # number of levels + final posterior
        self.posterior_abus = np.zeros((N, self.numberParam))
        self.posterior_mean = np.zeros(self.numberParam)
        self.posterior_std = np.zeros(self.numberParam)

        for i in range(len(self.posterior_abus[0, :])):
            self.posterior_abus[:, i] = samplesX[-1][:, i]
            self.posterior_mean[i] = np.mean(self.posterior_abus[:, i])
            self.posterior_std[i] = np.std(self.posterior_abus[:,i])

    def posteriorAnalysis(self, confidence=90):
        confidenceLow = (100-confidence)/2
        confidenceHigh = 100-confidenceLow

        xPostPredictive = np.linspace(np.min(self.x), np.max(self.x), 100)
        posteriorModel = np.zeros((len(self.posterior_abus[:, 0]), len(xPostPredictive)))
        postPredict = np.zeros((len(self.posterior_abus[:, 0]), len(xPostPredictive)))

        for i in range(len(posteriorModel[:, 0])):
            posteriorModel[i, :] = self.function(xPostPredictive, self.posterior_abus[i, :])

            if self.correlated_errors == False:
                postPredict[i, :] = posteriorModel[i, :] + norm.rvs(loc=0, scale=self.posterior_abus[i,-1],
                                                                    size=len(posteriorModel[i, :]))
            elif self.correlated_errors == True:
                cov = self.var_measurement(self.posterior_abus[i, -3], xPostPredictive) + self.posterior_abus[i, -1] * self.var_model(self.posterior_abus[i, -2], xPostPredictive)
                postPredict[i, :] = mvn.rvs(mean=posteriorModel[i, :], cov=(self.var_measurement(self.posterior_abus[i, -3], xPostPredictive) + self.posterior_abus[i, -1]*self.var_model(self.posterior_abus[i, -2], xPostPredictive)))

        self.upper = np.zeros(len(xPostPredictive))
        self.upper_model = np.zeros(len(xPostPredictive))
        self.mean = np.zeros(len(xPostPredictive))
        self.mean_model = np.zeros(len(xPostPredictive))
        self.lower = np.zeros(len(xPostPredictive))
        self.lower_model = np.zeros(len(xPostPredictive))

        for i in range(len(xPostPredictive)):
            
            ###################################################################
            #THE DIFFERENCE BETWEEN THE TWOS IS THAT PREDICT INCLUDES THE ERROR? YES!
            ####################################################################
            
            
            self.lower[i] = np.nanpercentile(postPredict[:, i], [confidenceLow])
            self.lower_model[i] = np.nanpercentile(posteriorModel[:, i], [confidenceLow])
            self.mean[i] = np.nanpercentile(postPredict[:, i], [50.])
            self.mean_model[i] = np.nanpercentile(posteriorModel[:, i], [50.])
            self.upper[i] = np.nanpercentile(postPredict[:, i], [confidenceHigh])
            self.upper_model[i] = np.nanpercentile(posteriorModel[:, i], [confidenceHigh])

        self.posteriorPredictive = postPredict
        self.xPostPredictive = xPostPredictive

    def residualAnalysis(self):
        # Generating the mean posterior for the x values
        posteriorModel = np.zeros((len(self.posterior_abus[:, 0]), len(self.x)))

        for i in range(len(posteriorModel[:, 0])):
            posteriorModel[i, :] = self.function(self.x, self.posterior_abus[i, :])

        self.meanModelResidual = np.zeros(len(self.x))

        for i in range(len(self.x)):
            self.meanModelResidual[i] = np.nanpercentile(posteriorModel[:, i], [50.])

        # Performing the residual analysis
        residual = self.y-self.meanModelResidual

        fig2, (ax2, ax3) = plt.subplots(1,2)
        ax2.scatter(self.x, residual)
        ax2.axhline(0)
        ax3.hist(residual, bins=30, density=True)
        x_plot_normal = np.linspace(min(residual), max(residual), 150)
        y_plot_normal = norm.pdf(x_plot_normal, 0, self.posterior_mean[-1])
        ax3.plot(x_plot_normal, y_plot_normal)

        print("\nRank correlation coefficient of the residuals is: ", spearmanr(residual, self.x)[0])

        # TODO Put in the official hypothesis test for acceptance of the normal distribution for the residuals

    def plot(self, realizations=False, N=5, y_axis='linear'):
        fig, ax = plt.subplots()

        # if y_axis == 'log':
        #     ax.set_yscale('log')
        #     ax.set_ylim(bottom=0.01)
        # else:
        #     ax.set_ylim(bottom=0, top=3.5)
        # ax.set_xlim([min(self.x), max(self.x)])
        
        #Plot the measured points    
        ax.scatter(self.x, self.y)
        # xPostPredictive is a vector of force values ranging between the experimental min and the experimental max
        ax.plot(self.xPostPredictive, self.mean_model, 'blue')

        #lower and upper delimit a 90% confidence interval
        ax.fill_between(self.xPostPredictive, self.lower, self.upper, color='blue', alpha=0.1, zorder=0,
                                label='Posterior predictive 90% credible interval')
        ax.fill_between(self.xPostPredictive, self.lower_model, self.upper_model, color='red', alpha=0.3, zorder=1,
                                label='Model posterior 90% credible interval')

        
      
        MM = 100
        post = np.zeros((MM,6))
        for i in range(MM):
            post[i,0] = self.xPostPredictive[i]
            post[i,1] = self.mean_model[i]
            post[i,2] = self.lower[i]
            post[i,3] = self.upper[i]
            post[i,4] = self.lower_model[i]
            post[i,5] = self.upper_model[i]
            
        print(i)
        file_name = "res_model.txt"
        np.savetxt(file_name, post, fmt="%.3f", delimiter=" ")

        ########################################################################
        #CAN I SAVE THESE VECTORS IN AN EXTERNAL FILE? YES!
        #SO THAT I CAN HAVE FIGURES THAT ARE THE SAME
        ########################################################################


        ax.set_xlabel('Average tangential stresses [kN/m2]')
        ax.set_ylabel('k [mm]')
        ax.legend()




        if realizations == True:
            plot_i = np.random.randint(0, len(self.posteriorPredictive[:, 0]), size=N)

            for i in range(N):
                ax.scatter(self.xPostPredictive, self.posteriorPredictive[plot_i[i], :])








