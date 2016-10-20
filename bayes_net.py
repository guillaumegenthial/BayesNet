import numpy as np
from scipy.special import gammaln
from collections import OrderedDict
import copy

class BayesNet(object):
	"""
	Class to store a bayes net and implement scoring functions

	Attributes:
		parents (OrderedDict): parents[i] is a set of the parents of i
		names_to_index (OrderedDict): names_to_index["age"] = 1 for instance
		n (int): number of nodes

	"""
	def __init__(self, names=[], bn=None):
		if bn is not None:
			self.copy(bn)
			
		else:
			self.parents = OrderedDict()
			self.names_to_index = OrderedDict()
			self.index_to_names = OrderedDict()
			self.n = 0

			if names != []:
				self.names_init(names)

	def names_init(self, names):
		"""
		initialize the names_to_index and index_to_names attributes
		initialize parents[i] = set() (no edges for the moment)

		Args:
			names (list of string): the names of the nodes

		"""
		self.names_to_index = {name: index for index, name in enumerate(names)}
		self.index_to_names = {index: name for index, name in enumerate(names)}
		self.n = len(self.names_to_index)
		for i in xrange(self.n):
			self.parents[i] = set()

	def random_init(self, max_parents=None):
		"""
		Add edges randomly

		For each node, pick a random number of the desired number of parents.
		Then, for each candidate, pick another random number. In average,
		the node will have the desired number of parents.
		"""
		if max_parents is None:
			max_parents = self.n - 1

		for i in xrange(self.n):
			nparents = np.random.randint(0,max_parents + 1)
			p = nparents / (self.n + 1.0)
			for j in xrange(self.n):
				if j!= i and np.random.uniform() < p:
					self.add_edge(j,i)


	def merge(self, g1, g2, p1=1, p2=1, max_parents=None, mut_rate=0):
		"""
		Picks edges from both g1 and g2 according to some random policy

		Args
			g1 (BayesNet)
			g1 (BayesNet)
			p1 (float in [0,1]): proba of an edge in g1 being in self
			p2 (float in [0,1]): proba of an edge in g2 being in self
				p1 + p2 = 1
			max_parents (int)

		"""
		# merge randomly the two graphs
		self.random_merge(g1, g2, p1, p2)

		# introduce mutations
		self.mutate(mut_rate)

		# remove extra parents
		self.remove_extra_parents(max_parents)

	def random_merge(self, g1, g2, p1, p2):
		"""
		Creates graph from edges both in g1 and g2.
		Adds edges according to proba p1 and p2
		"""
		for i, js in g1.parents.iteritems():
			for j in js:
				if np.random.uniform() < p1 or p1 == 1:
					self.add_edge(j, i)
		for i, js in g2.parents.iteritems():
			for j in js:
				if np.random.uniform() < p2 or p2 == 1:
					self.add_edge(j, i)

	def mutate(self, mut_rate=0):
		"""
		Introduces new edges with a probability mut_rate

		Args
			mut_rate (float in [0,1]): proba of mutation
		"""
		if mut_rate != 0:
			for i in xrange(self.n):
				for j in xrange(self.n):
					p = np.random.uniform()
					if p < mut_rate:
						if p<mut_rate/2:
							self.add_edge(i, j)
						else:
							self.remove_edge(i, j)

	def remove_extra_parents(self, max_parents=None):
		"""
		Removes extra edges if does not respect max parents constraint

		Picks edges randomly
		"""
		if max_parents is not None:
			for i, js in self.parents.iteritems():
				if len(js) > max_parents:
					indices = np.random.permutation(range(len(js)))
					for j in indices[0:len(js)-max_parents]:
						self.remove_edge(j, i)

	def plot(self, mode="pretty"):
		"""
		Print the network in the terminal

		Args
			mode (string): if "pretty", will print the names
				of the nodes, else just the indices
		"""
		if mode == "pretty":
			for ci, ps in self.parents.iteritems():
				for pi in ps:
					p = self.index_to_names.get(pi)
					c = self.index_to_names.get(ci)
					print "{} => {}".format(p, c)
		else:
			for k, v in self.parents.iteritems():
				print "[{}] <= {}".format(k, list(v))

	def export(self, file_name):
		"""
		Uses graphviz to produce a png image of the network

		Args
			file_name (strint): path of the save location
		"""
		try:
			import graphviz as gv
			g = gv.Digraph(format='png')
			for name in self.names_to_index.keys():
				g.node(name)
			for ci, ps in self.parents.iteritems():
				for pi in ps:
					p = self.index_to_names.get(pi)
					c = self.index_to_names.get(ci)
					g.edge(p, c)
			g.render(file_name)
		except Exception, e:
			print "Unable to load graphviz"

	def save(self, file_name):
		"""
		Saves the graph in the desired format

		Example
			parent1, child1
			parent2, child2
		"""
		with open(file_name, "w") as f:
			for child_index, parents in self.parents.iteritems():
				for parent_index in parents:
					parent = self.index_to_names.get(parent_index)
					child = self.index_to_names.get(child_index)
					f.write("{},{}\n".format(parent, child))

	def load(self, file_name):
		"""
		Loads structure from file. See save method
		"""
		with open(file_name) as f:
			for line in f:
				line = line.strip().split(',')
				if len(line) == 2:
					p, c = line[0], line[1]
					p_index, c_index = self.names_to_index[p], self.names_to_index[c]
					self.add_edge(p_index, c_index)

	def is_cyclic(self):
		"""
		Returns True if a cycle is found else False.

		Iterates over the nodes to find all the parents' parents, etc.
		A cycle is found if a node belongs to its own parent's set.
		"""
		all_parents = copy.deepcopy(self.parents)
		update = True
		while update:
			update = False
			for i in xrange(self.n):
				parents = list(all_parents[i])
				nparents = len(parents)
				for p in parents:
					all_parents[i].update(all_parents[p])
				if nparents != len(all_parents[i]):
					update = True
				if i in all_parents[i]:
					return True

		return False

	def copy(self, bn):
		"""
		Copies the structure of bn inside self and erases everything else

		Args
			bn (BayesNet): model
		"""
		self.index_to_names = copy.deepcopy(bn.index_to_names)
		self.names_to_index = copy.deepcopy(bn.names_to_index)
		self.n = copy.deepcopy(bn.n)
		self.parents = copy.deepcopy(bn.parents)


	def add_edge(self, parent, child, max_parents=None):
		"""
		Adds edge if respects max parents constraint and does not create a cycle

		Args
			parent (int): id of parent
			child (int): id of child
			max_parents (int): if None, no constraints

		Returns
			add_it (bool): True if actually added the edge
		"""

		add_it = False
		if max_parents is not None:
			if len(self.parents[child]) < max_parents:
				add_it = True
		else:
			add_it = True

		self.parents[child].add(parent)
		if self.is_cyclic():
			self.remove_edge(parent, child)
			add_it = False

		return add_it

	def remove_edge(self, parent, child):
		try:
			self.parents[child].remove(parent)
		except Exception, e:
			pass

	def score(self, data):
		"""
		Computes bayesian score of the structure given some data
		assuming uniform prior

		Args
			data (np array): (nsamples, nfeatures)

		Returns
			s (float): bayesian score

		Example
			s = bn.score(data)
		"""
		s = 0
		r = self.compute_r(data)
		for i in xrange(self.n):
			s += self.score_node(i, data, r)
		return s

	def compute_r(self, data):
		"""
		Computes the number of instantiations of each node

		Args
			data (np array): (nsamples, nfeatures)
		Returns
			r (OrderedDict): r[i] = r_i
		"""
		r = OrderedDict()
		for i in xrange(self.n):
			r[i] = np.unique(data[:,i]).shape[0]
		return r

	def score_node(self, i, data, r):
		"""
		Computes \sum_{j=1}^{q_i} \ln(\frac{\Gamma(\alpha_{ij0})}{\Gamma(\alpha_{ij0}+m_{ij0})}) 
		+ \sum_{k}^{r_i} \ln(\frac{\Gamma(\alpha_{ijk} + m_{ijk})}{\Gamma(\alpha_{ijk})})

		Args
			i (int): node
			data (np array): (nsamples, nfeatures)
			r (dict of np array): r[i] = nb possible instances of i
		Returns
			s (float): contribution to log score of node i
		"""
		m = OrderedDict()
		m0 = OrderedDict()
		columns = [i] + list(self.parents.get(i))
		extracted_data = data[:, columns]

		# counting nb of each instance of (node, parents) and (parents)
		for sample in extracted_data:
			t = tuple(sample)
			if t in m:
				m[t] += 1
			else:
				m[t] = 1
			t0 = tuple(sample[1:])
			if t0 in m0:
				m0[t0] += 1
			else:
				m0[t0] = 1

		# Adding contribution to the score (assuming uniform prior)
		s = 0.
		for c in m0.itervalues():
			s -= gammaln(r[i] + c)
			s += gammaln(r[i])
		for c in m.itervalues():
			s += gammaln(1 + c)

		return s