import numpy as np
from scipy.special import gammaln
from collections import OrderedDict
import sys
import time

from bayes_net import BayesNet
try:
	from plotting import PlotManager
except Exception, e:
	print "Unable to import plotting"

class Strategy(object):
	"""
	General class to implement different structure learning methods

	Attributes
		random_explore (bool): if True, gives random advantage to visited BN
		random_method (string): how to update the margin of exploration
			choices:
				- "momentum": no need to specify margin, computed from past epochs
				- "adam": need to specify margin, updates on the run
		margin (float): how big the jump in exploration will be
		epsilon (float): margin of exploration
		beta (float): in [0,1], update rate of the progress
		gamma (float): in [0,1], update rate of the square of the progress
	"""
	def __init__(self, **kwargs):
		self.strategy_name = kwargs.get("strategy_name", "hill_climbing")
		self.random_explore = kwargs.get("random_explore", False)
		self.random_method = kwargs.get("random_method", "adam")
		self.margin = kwargs.get("margin", 50)
		self.beta = kwargs.get("beta", 0.9)
		self.gamma = kwargs.get("gamma", 0.99)
		self.plotting = kwargs.get("plotting", False)
		if self.random_explore:
			self.epsilon = self.margin
		else:
			self.epsilon = 0

		self.step = 0
		self.m = 0
		self.v = 0

		self.start_no = 0
		if self.plotting:
			try:
				self.plt_mgr = PlotManager(title="Score")
			except Exception, e:
				print "Error while loading Plot Manager"

	def run(self, **kwargs):
		"""
		Run the actual strategy

		Args
			**kwargs (dict): dictionnary with method specific args
		"""
		t0 = time.time()
		strategy = self.strategy_name
		if strategy == "hill_climbing":
			g, s = self.hill_climbing(**kwargs)
		elif strategy == "multi_start_hill_climbing":
			g, s = self.multi_start_hill_climbing(**kwargs)
		elif strategy == "brute_force":
			g, s = self.brute_force(**kwargs)
		elif strategy == "genetic":
			g, s = self.genetic(**kwargs)
		elif strategy == "memetic":
			g, s = self.genetic(local_search=True, **kwargs)
		elif strategy == "k2":
			g, s = self.k2(**kwargs)
		elif strategy == "multi_k2":
			g, s = self.multi_k2(**kwargs)
		else:
			print "Strategy {} is unknown".format(strategy)
			raise

		t1 = time.time()
		print "Best score is {}".format(s)
		print "Time elapsed is {}".format(t1 - t0)
		
		if self.plotting:
			try:
				self.plt_mgr.close()
			except Exception, e:
				pass
		return g, s
		
	def update_epsilon(self, s, s_new):
		"""
		Updates the exploration margin as an EMA of the progress

		Args
			s (float): last candidate score
			s_new (float): new candidate score
		"""
		if self.random_explore:
			method = self.random_method
			if method == "adam":
				self.adam(s, s_new)
			elif method == "momentum":
				self.momentum(s, s_new)
			else:
				print "Unknow update method for epsilon"
				raise
			self.step += 1
			if self.plotting:
				try:
					self.plt_mgr.add(name="epsilon", y=self.epsilon)
					self.plt_mgr.update()
				except Exception,e:
					pass
			print "New margin for random exploration is {}".format(self.epsilon)


	def adam(self, s, s_new):
		"""
		Updates epsilon depending on 
			- a margin 
			- history of progress
			- the square of progress
		"""
		eps = self.epsilon
		m = self.m
		v = self.v
		beta = self.beta
		gamma = self.gamma
		m_new = np.absolute(s_new - s)
		v_new = np.square(s_new-s)
		if self.step > 0:
			self.m = beta*m + (1 - beta)*m_new
			self.v = gamma*v + (1 - gamma)*v_new
		else:
			self.m = m_new
			self.v = v_new
		self.epsilon = self.margin * self.m / np.sqrt(self.v)

	def momentum(self, s, s_new):
		"""
		Updates epsilon depending on the history of progress
		"""
		eps = self.epsilon
		m = self.m
		beta = self.beta
		m_new = np.absolute(s_new - s)
		if self.step > 0:
			self.m = beta*m + (1 - beta)*m_new
		else:
			self.m = m_new
		self.epsilon = self.m

	def brute_force(self, **kwargs):
		"""
		Sample random bayesian network and keep the best

		Args
			names (list of string): the names of the nodes
			data (np array): (nsamples, nfeatures)
		"""
		# get args
		names = kwargs.get("names")
		data = kwargs.get("data")
		nsamples = kwargs.get("nsamples", 1000)

		# initialize
		g = BayesNet(names)
		g.random_init()
		s = g.score(data)

		# explore
		for i in xrange(nsamples):
			sys.stdout.write("\rIter {}".format(i))
			sys.stdout.flush()
			g_new = BayesNet(names)
			g_new.random_init()
			s_new = g_new.score(data)
			if s_new > s:
				print "\nFound new best score at {}".format(s_new)
				g = g_new
				s = s_new
		return g, s

	def best_neighbour(self, names, data, g0, max_parents):
		"""
		Find best neighboor of a BN

		Args
			names (list of string): the name of the nodes
			data (np array): (nsamples, nfeatures)
			g0 (BayesNet): the reference

		Returns
			g: best neighbour
			s: score of best neighbour

		"""
		print "Searching for best neighbour"
		# reference variables
		n = g0.n
		r = g0.compute_r(data)
		s0 = g0.score(data)

		# best candidate so far
		g = BayesNet(bn=g0)
		s = s0
		s_eps = s0
		found_new = False

		# working graph
		g_work = BayesNet(bn=g0)
		if max_parents is None:
			max_parents = n - 1


		def update_best(mode="add"):
			"""
			When called, evaluate the working graph and update best candidate
			The s update must take place out of the function scope for python limitations
			"""
			# if mode == "rem" or not g_work.is_cyclic():
			s_new = s0 - s_i + g_work.score_node(i, data, r)
			# we give a random advantage to the candidate based on previous updates
			s_eps_new = s_new + self.epsilon * np.random.rand()
			if s_eps_new > s_eps:
				print "Found new candidate ({}) at {}".format(mode, s_new)
				g.copy(g_work)
				return s_new, s_eps_new, True

			return s, s_eps, found_new

		# iterate over node center of the modification
		for i in xrange(n):
			parents = g0.parents[i]
			s_i = g0.score_node(i, data, r)
			# 1. remove edge
			for j in parents:
				g_work.remove_edge(j,i)
				s, s_eps, found_new = update_best("rem")
				g_work.add_edge(j,i)

			# 2. add edge
			if len(parents) < max_parents:
				for j_prime in xrange(n):
					if j_prime not in parents:
						if g_work.add_edge(j_prime, i):
							s, s_eps, found_new = update_best("add")
							g_work.remove_edge(j_prime, i)

			# 3. reverse direction
			for j in parents:
				if len(g0.parents[j]) < max_parents:
					g_work.remove_edge(j,i)
					if g_work.add_edge(i,j):
						s, s_eps, found_new = update_best("rev")
						g_work.remove_edge(i,j)
					g_work.add_edge(j,i)

		self.update_epsilon(s0, s)
		return g, s, found_new

	def hill_climbing(self, **kwargs):
		"""
		Implements Hill Climbing Algorithm

		Args
			names (list of string): the name of the nodes
			data (np array): (nsamples, nfeatures)
			max_iter (int): max number of iteration
			g0 (BayesNet): the start point

		Returns
			g: best graph found
			s: score of best graph

		"""
		# get args
		names = kwargs.get("names")
		data = kwargs.get("data")
		max_iter = kwargs.get("max_iter", 20)
		max_parents = kwargs.get("max_parents", None)

		# initialize
		g0 = BayesNet(names)
		g0.random_init(max_parents=max_parents)
		g = g0
		s = g0.score(data)
		found_new = True
		niter = 0

		# explore
		while found_new and niter < max_iter:
			print "Iter {}".format(niter)
			niter += 1
			g, s, found_new = self.best_neighbour(names, data, g, max_parents)
			if self.plotting:
				try:
					self.plt_mgr.add(name="score hill climbing {}".format(self.start_no), y=s)
					self.plt_mgr.update()
				except Exception, e:
					pass
		return g, s

	def multi_start_hill_climbing(self, **kwargs):
		"""
		Executes Hill Climbing from multiple starting points
		Args
			names (list of string): the name of the nodes
			data (np array): (nsamples, nfeatures)
			max_iter (int): max number of each iteration
			nb_start (int): number of starting points

		Returns
			g: best graph found
			s: score of best graph

		"""
		# args
		names = kwargs.get("names")
		data = kwargs.get("data")
		max_iter = kwargs.get("max_iter", 30)
		nb_start = kwargs.get("nb_start", 3)
		max_parents = kwargs.get("max_parents", None)

		# explore
		print "-"*5 + " Hill Climbing no 1 "+ "-"*5
		g, s = self.hill_climbing(
			names=names, 
			data=data, 
			max_iter=max_iter, 
			max_parents=max_parents)
		for i in xrange(nb_start-1):
			print "-"*5 + " Hill Climbing no {} ".format(i+2) + "-"*5
			self.step = 0
			self.start_no += 1
			g_new, s_new = self.hill_climbing(
				names=names, 
				data=data, 
				max_iter=max_iter, 
				max_parents=max_parents)
			if s_new > s:
				print "Found new best multistart score at {}".format(s_new)
				g, s = g_new, s_new

		return g, s

	def best_parent(self, g, s, i, data, max_parents):
		"""
		Returns g by adding to node i the best parent that maximizes the score
		"""
		found_new = False
		r = g.compute_r(data)
		s_i = g.score_node(i, data, r)
		s_max = s
		g_max = g

		g_work = BayesNet(bn=g)
		for j in range(g.n):
			if j not in g_work.parents[i]:
				success = g_work.add_edge(j, i, max_parents)
				if success:
					s_new = s - s_i + g_work.score_node(i, data, r)
					if s_new > s_max:
						found_new = True
						s_max = s_new
						g_max = BayesNet(bn=g_work)
				g_work.remove_edge(j, i)

		return g_max, s_max, found_new



	def k2(self, **kwargs):
		"""
		Implements k2 algorithm
		"""
		names = kwargs.get("names")
		data = kwargs.get("data")
		max_iter = kwargs.get("max_iter", 30)
		nb_start = kwargs.get("nb_start", 3)
		max_parents = kwargs.get("max_parents", None)

		ordering = np.random.permutation(range(len(names)))
		g = BayesNet(names)
		s = g.score(data)

		for i in ordering:
			found_new = True
			while found_new:
				print "Node {}, score is {}".format(i, s)
				g, s, found_new = self.best_parent(g, s, i, data, max_parents)
				if self.plotting:
					try:
						self.plt_mgr.add(name="score k2 {}".format(self.start_no), y=s)
						self.plt_mgr.update()
					except Exception, e:
						pass

		return g, s

	def multi_k2(self, **kwargs):
		"""
		Run k2 a nb_start times and keep the best network
		"""
		nb_start = kwargs.get("nb_start", 6)
		g, s = None, None
		for i in xrange(nb_start):
			print "-"*5 + " K2 no {} ".format(i)+ "-"*5
			g_new, s_new = self.k2(plot=False, **kwargs)
			if s is None or s_new > s:
				g, s = g_new, s_new
				if self.plotting:
					try:
						self.plt_mgr.add(name="score multi k2", y=s)
						self.plt_mgr.update()
					except Exception, e:
						pass
			self.start_no += 1
		return g, s


	def evolve(self, names, data, population, max_parents, mut_rate, max_pop, local_search):
		"""
		Given a population, creates a new population with random pairing and mixing

		If local seach is true, children is the best neigbour of the random merge
		"""
		new_population = []
		s_tot = sum([s for (_, s) in population ])
		n = len(population)
		population = np.random.permutation(population)
		for p in xrange(n/2):
			(g1, s1) = population[2*p]
			(g2, s2) = population[2*p + 1]
			nchildren = int(n*(s1 + s2)/s_tot) + 1
			for i in xrange(nchildren):
				if len(new_population) < max_pop:
					g = BayesNet(names)
					g.merge(g1, g2, s1/s_tot, s2/s_tot, max_parents, mut_rate)
					if local_search:
						g, s, _ = self.best_neighbour(names, data, g, max_parents)
					else:
						s = g.score(data)
					new_population += [(g, s)]
					if self.plotting:
						try:
							self.plt_mgr.add(name="Genetic Score", y=s)
							self.plt_mgr.update()
						except Exception, e:
							pass

		return new_population


	def genetic(self, **kwargs):
		"""
		Implements genetic reproduction

		If local search is set to True, implements mimetic
		"""
		names = kwargs.get("names")
		data = kwargs.get("data")
		max_iter = kwargs.get("max_iter", 30)
		nb_start = kwargs.get("nb_start", 10)
		max_pop = kwargs.get("max_pop", nb_start)
		max_parents = kwargs.get("max_parents", None)
		mut_rate = kwargs.get("mut_rate", 0.01)
		local_search = kwargs.get("local_search", False)

		# initialize the population
		s_max = None
		g_max = None
		population = []
		for i in xrange(nb_start):
			g = BayesNet(names)
			g.random_init(max_parents)
			if local_search:
				g, s, _ = self.best_neighbour(names, data, g, max_parents)
			else:
				s = g.score(data)

			population += [(g,s)]
			if s > s_max or s_max is None:
				s_max = s
				g_max = g

		# let evolution do its work
		criteria = True
		niter = 0

		def update_criteria_from(population):
			s = None
			g = None
			for (_g, _s) in population:
				if s is None or _s > s:
					s = _s
					g = _g
			if s > s_max:
				return g, s, True
			else:
				return g_max, s_max, True


		while criteria and niter < max_iter:
			print "Iter {}, Population {}".format(niter, len(population)) 
			population = self.evolve(names, data, population, max_parents, mut_rate, max_pop, local_search)
			g_max, s_max, criteria = update_criteria_from(population)
			if self.plotting:
				try:
					self.plt_mgr.add(name="Genetic Score Max", y=s_max)
					self.plt_mgr.update()
				except Exception, e:
					pass
			niter += 1

		return g_max, s_max


