import numpy as np 
import scipy.stats as ss 

def genDist(dictionary):

	variables      = list(dictionary.keys())
	n_params       = len(variables)
	new_dictionary = {}

	for i in range(n_params):

		if dictionary[variables[i]][-1] == 'uniform':

			new_dictionary[i] = ss.uniform(dictionary[variables[i]][0], dictionary[variables[i]][1])

		elif dictionary[variables[i]][-1] == 'normal':

			new_dictionary[i] = ss.norm(dictionary[variables[i]][0], dictionary[variables[i]][1])			

	return new_dictionary

'''
np.seterr(all = 'ignore')
np.set_printoptions(precision=3, suppress = True)

# -------------- Example --------------
# Current implementation using normal and uniform distributions only
# Note about using uniform: first element is the minimum of the bound
# Meaning, [0.5, 0.4] implies 0.5<x<0.9
# ALWAYS use ordered dictionary to avoid variables swapping with each other (in HMC)
# ------------------------------------------------------
from collections import OrderedDict
params              = OrderedDict()
params['h']         = [0.5, 0.4, 'uniform']
params['omega_cdm'] = [0.2, 0.1, 'normal']
params['omega_de']  = [0.7, 0.01, 'normal']
# ------------------------------------------------------

distributions = genDist(params)

#samps = distributions[0].rvs(3)
#print distributions[0].pdf(samps)

print distributions[0].rvs(3)
print distributions[1].rvs(3)
print distributions[2].rvs(3)
'''













