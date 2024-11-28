import numpy as np
import BayesianLib as bayes
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
plt.close('all')





#################
# Measured data #
#################

# Applied tangential stresses (in kPa)
tau = np.array([157.4, 216.5, 275.5, 314.9, 354.2, 393.6])

# 'Failure' stress determined by the standard approach (in kPa)
fail_stress = 393

# Measured creep (in mm)
ks  = np.array([0.21 , 0.24,  0.48,  0.66,  0.94,  1.49])


#######################
# Defining the priors #
#######################
prior_Ea = ['Lognormal', 2000., 400.] #in kPa/mm
prior_taua = ['Lognormal', 300, 100] #in kPa
prior_ksmin = ['Lognormal', 0.2, 0.1] #in mm
prior_std = ['Uniform', 0, 1]  #in mm


prior = [prior_Ea, prior_taua, prior_ksmin, prior_std]


###############################
# Defining the model function #
###############################

anchor_hyperbolic = lambda x, param: 1/param[0]*x/(1-x/param[1])+param[2]


#######################################################################
# Performing the analysis: determining the model parameter posteriors #
#######################################################################

regression = bayes.BayesianRegression(anchor_hyperbolic, 4, correlated_errors=False)   # Initialising object, number of parameters
regression.addData(tau, ks)   # Adding the data
posterior = regression.regression(prior)    # Performing the regression
posteriorPredictive = regression.posteriorAnalysis()   # Performing posterior predictive analysis
plot = regression.plot()   # Plotting the different aspects.


##############################################
# Calculation of anchor characteristic force #
##############################################

# Definition of a vector of values to be introduced into the model to calculate failure
x_predict = np.linspace(0.7*max(tau), 1.3*max(tau), 250)
failure_force = np.zeros(regression.N)

for i in range(regression.N):
    #IS THIS LINE TO INSERT THE POSTERIOR DISTRIBUTION INTO THE MODEL? YES!
    pred_model = anchor_hyperbolic(x_predict, regression.posterior_abus[i, :])
    #I add the uncertainties on the residuals
    post_pred = pred_model + norm.rvs(loc=0, scale=regression.posterior_abus[i, 3], size=len(pred_model))

    for j in range(1, len(x_predict)):
        if post_pred[j] >= 2 and post_pred[j-1]<2: #failure criterion: current step greater than 2mm, but previous lower than 2mm
            failure_force[i] = x_predict[j]
            break
        else:
            failure_force[i] = np.nan


############################
# Output: model parameters #
############################
print('Mean of model parameters is: ', regression.posterior_mean)
print('Std of model parameters is: ', regression.posterior_std)



##############################################################
#Output : histogram of failure force including 5% percentile #
##############################################################
plt.figure()
plt.hist(failure_force[~np.isnan(failure_force)],density=True, bins=40)

#Calculate the 5% percentile
char_force = np.nanpercentile(failure_force, 5)
print('The anchor characteristic strenght is equal to: ',char_force,'kPa')
plt.axvline(char_force, color='black', label=round(char_force, 3))



plt.axvline(fail_stress, color='red', label='test ' + str(fail_stress) + 'kPa')

plt.xlabel('Average tangential stresses [kPa]')
plt.ylabel('Probability')
plt.legend()

post = regression.posterior_abus[:,:]





