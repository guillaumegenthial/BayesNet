import sys
import time
import collections
import numpy as np
try:
	import pyqtgraph as pg
	from pyqtgraph.Qt import QtGui, QtCore
	pg.setConfigOption('background', 'w')
except Exception, e:
	print "Unable to import pyqtgraph"
	raise

class LivePlotter(object):
	def __init__(self, **kwargs):

		self.name = kwargs.get("name", "live_plotter")
		self.frequency = kwargs.get("frequency", 0.1)
		self.downsample = kwargs.get("downsample", 10)
		self.point_nb = kwargs.get("point_nb", 100)
		self.size = kwargs.get("size", (600, 300))
		self.pen = kwargs.get("pen", "r")
		self.x_axis = kwargs.get("x_axis", "x")
		self.y_axis = kwargs.get("y_axis", "y")
		self.x_unit = kwargs.get("x_unit", "t")
		self.y_unit = kwargs.get("y_unit", "")

		self.last_refresh = time.time()

		self.x, self.y = [], []

		self.win = kwargs.get("win", pg.GraphicsWindow().resize(self.size[0], self.size[1]))

		self.p = self.win.addPlot(title=self.name)
		self.p.setLabel('left', self.y_axis, units=self.y_unit)
		self.p.setLabel('bottom', self.x_axis, units=self.x_unit)
		self.plot = self.p.plot(self.x, self.y, pen=self.pen)


	def add(self, y, x=None):
		if x is None:
			x = time.time()
		self.x += [x]
		self.y += [y]

	def update(self):
		t = time.time()
		frequency = self.frequency
		last_refresh = self.last_refresh
		downsample = self.downsample
		point_nb = self.point_nb

		if frequency is not None:
			if t - last_refresh < frequency:
				return

		if point_nb is not None:
			x_size = len(self.x)
			downsample = x_size / point_nb
		
		# self.plot.setData(self.x, self.y, downsample=downsample)
		self.plot.setData(self.x, self.y)
		pg.QtGui.QApplication.processEvents()
		self.last_refresh = t

	def close(self):
		self.win.close()

class PlotManager(object):
	def __init__(self, **kwargs):
		self.title = kwargs.get("title", "Plots")
		self.size = kwargs.get("size", (800, 400))
		self.nline = kwargs.get("nline", 3)
		self.frequency = kwargs.get("frequency", 0.1)
		self.downsample = kwargs.get("downsample", 10)
		self.point_nb = kwargs.get("point_nb", 100)
		self.nplots = -1


		self.plots = collections.OrderedDict()
		self.win = pg.GraphicsWindow(title=self.title)
		self.win.resize(self.size[0], self.size[1])

	def add(self, name, y, x=None, **kwargs):
		if name not in self.plots:
			self.nplots += 1
			if self.nplots % self.nline == 0:
				self.win.nextRow()

			self.plots[name] = LivePlotter(
				name=name, 
				win=self.win, 
				frequency=self.frequency,
				downsample=self.downsample,
				point_nb=self.point_nb,
				**kwargs)

		self.plots[name].add(y, x)

	def update(self):
		for name, plot in self.plots.iteritems():
			plot.update()

	def close(self):
		try:
			wait = input("Press ENTER to close plots")
		except Exception, e:
			pass
		for name, plot in self.plots.iteritems():
			plot.close()


# length = 10000
# costs  = np.arange(length)

# plt_mgr = PlotManager(
# 	title="plots", 
# 	nline=2,
# 	frequency=0.5,
# 	point_nb=10,
# 	downsample=None)

# for i in range(length):
# 	cost = costs[i]
# 	plt_mgr.add("cost", cost)
# 	plt_mgr.add("time", time.time())
# 	plt_mgr.add("time2", time.time())
# 	plt_mgr.update()

# plt_mgr.close()


