# Script ot generate Latin Hypercube Samples with lhs routine 
# Can further specify other methods - see lhs.pdf manual 
# n is the number of points we require 
# d is the dimensionality of the problem 

setwd('your_working_directory')
library(lhs)
n = 1000
d = 8
X = maximinLHS(n, d)
write.csv(X, 'lhs_samples/samples_lhs_8d_1000.csv')

