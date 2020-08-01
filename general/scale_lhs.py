import pandas as pd 
import numpy as np 

'''
Note that for this project, we have been using the KiDs-450 prior to scale the training points 
But a more conservative approach would be to scale the training points to the high density region only 
'''

# scaling according to Kids-450 prior 
min_max_prior = np.array([[0.010, 0.400],
	[0.019, 0.026],
	[1.700, 5.000],
	[0.700, 1.300],
	[0.640, 0.820],
	[0.000, 2.0],
	[0.060, 1.00],
	[-6.00, 6.00]])

samples  = pd.read_csv(directory+'lhs_samples/samples_lhs_8d_1000.csv').iloc[:,1:]
samples_ = np.array(samples)
scaled   = min_max_prior[:,0] + (min_max_prior[:,1] - min_max_prior[:,0])*samples_ 

# np.savetxt('training_points/input_parameters.txt', scaled)
