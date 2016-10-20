import numpy as np
from scipy.special import gammaln
from collections import OrderedDict
import copy
import sys
from optparse import OptionParser

from bayes_net import BayesNet
from strategy import Strategy

def load_data(file_name):
	"""
	Load data from a csv file

	Returns
		names (list of string): the names of the columns
		samples (np array): (nsamples, nfeatures)
	"""
	samples = []
	with open(file_name) as f:
		for index, line in enumerate(f):
			if index==0:
				names = map(lambda s: s.strip('"'), line.strip().split(','))
			else:
				sample = map(int, line.split(','))
				samples.append(sample)

	return names, np.asarray(samples)


def args():
	parser = OptionParser(usage='usage: %prog [options] ')
	parser.add_option('-b', '--base',
                      type='choice',
                      action='store',
                      dest='base',
                      choices=['titanic', 'whitewine', 'schoolgrades',],
                      default='titanic',
                      help='Name of the base',)
	parser.add_option('-s', '--strategy',
                      type='choice',
                      action='store',
                      dest='strategy',
                      choices=['hill_climbing', 'multi_start_hill_climbing', 'k2', 'multi_k2', 'genetic', 'memetic',],
                      default='hill_climbing',
                      help='Strategy',)
	parser.add_option('-r', '--random',
                      type='choice',
                      action='store',
                      dest='random_explore',
                      choices=[True, False],
                      default=True,
                      help='Use random exploration',)
	parser.add_option('-m', '--method',
                      type='choice',
                      action='store',
                      dest='random_method',
                      choices=['adam', 'momentum',],
                      default='adam',
                      help='Method to use for random exploration',)
	parser.add_option("-i", "--iter",
                      action="store",
                      dest="max_iter",
                      type="int",
                      default=40,
                      help="Maximum of iteration for strategy")
	parser.add_option("-n", "--nstart",
                      action="store",
                      type="int",
                      dest="nb_start",
                      default=20,
                      help="Number of start for multi start strategies")
	parser.add_option("-o", "--offset",
                      action="store",
                      dest="margin",
                      type="int",
                      default=50,
                      help="Parameter for random exploration (method adam)")
	parser.add_option("-e", "--evolution",
                      action="store",
                      dest="mutation_rate",
                      type="float",
                      default=0.01,
                      help="Mutation rate for evolution, between 0 and 1")
	parser.add_option("-p", "--parents",
                      action="store",
                      type="int",
                      dest="max_parents",
                      default=None,
                      help="Maximum number of parents for a node")
	parser.add_option("-g", "--graph",
					  type="choice",
                      action="store",
                      dest="plotting",
                      choices=[True, False],
                      default=True,
                      help="Maximum number of parents for a node")

	(options, _) = parser.parse_args()
	return options

options = args()

# 1. Loading data
DATA_PATH = "data"
SAVE_PATH = "result"
IMG_PATH = "img"

# 2. Using a strategy
MAX_PARENTS = options.max_parents
MAX_ITER = options.max_iter
NB_START = options.nb_start
STRATEGY = options.strategy
RANDOM_EXPLORE = options.random_explore
RANDOM_METHOD = options.random_method
MUT_RATE = options.mutation_rate
MARGIN = options.margin
PLOT = options.plotting
BASE = options.base

names, data = load_data(DATA_PATH + "/{}.csv".format(BASE))

S = Strategy(
	strategy_name=STRATEGY, 
	random_explore=RANDOM_EXPLORE, 
	random_method=RANDOM_METHOD, 
	margin=MARGIN,
	plotting=PLOT)

g, s = S.run(
	names=names, 
	data=data, 
	max_iter=MAX_ITER,
	nb_start=NB_START,
	max_parents=MAX_PARENTS,
	mut_rate=MUT_RATE)

# 3. Save graph and export
g.save(SAVE_PATH + "/{}-{}-{}-{}-{}.gph".format(BASE, s, STRATEGY, RANDOM_EXPLORE, MUT_RATE))
g.export(IMG_PATH + "/{}-{}-{}-{}-{}.gph".format(BASE, s, STRATEGY, RANDOM_EXPLORE, MUT_RATE))

# 4. Load a pre existing model from file
# g = BayesNet(names)
# g.load("{}.gph".format(BASE))

