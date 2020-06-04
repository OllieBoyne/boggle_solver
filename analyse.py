"""Functions for performing analysis on boggle boards"""

from boggle import Boggle
from solver import Solver
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm

def run(n=1, board_id=None):
	"""
	Solves several generated boggle configs

	:param method: solving method
	:param n: number of boards to run
	:param board: name of board to use
	:return: n board layouts, n heatmaps by points, n lists of worded solutions
	"""

	if board_id is None:
		board = Boggle()
	else:
		board = Boggle(board=board_id)

	board.width = 5

	layouts = []
	solutions = []

	solver = Solver()

	points = np.zeros((n, board.width, board.width))

	with tqdm(np.arange(n)) as tqdm_iterator:
		for i in tqdm_iterator:
			lay = board.gen()
			layouts.append(lay)

			sol = solver.solve(lay)
			solutions.append(sol)

			# add all points to heatmap
			for word in sol:
				val = len(word) - 3
				for route in sol[word]:
					for (x,y) in route:
						points[i, y, x] += val

	return layouts, points, solutions


def board_heatmap(ax, data, cmap="jet"):
	norm = colors.Normalize(vmin=0, vmax=data.max())
	ax.imshow(data, cmap=cmap, norm=norm)
	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	plt.colorbar(sm, ax = ax)

def point_distribution():
	"""Run experiment to measure distribution of points across board"""

	n = 1000
	layouts, points, solutions = run(n=n)

	means = points.mean(axis=0)
	vars = np.std(points, axis=0)

	for i in range(1000):
		if (points[i] == 0 ).any():
			print(layouts[i])
			print(points[i])
			raise ValueError

	inactive = np.mean(points == 0, axis=0)

	cmap = "jet"

	fig, (ax_means, ax_std, ax_inactive) = plt.subplots(ncols=3)

	## plot means
	board_heatmap(ax_means, means)
	board_heatmap(ax_std, vars)
	board_heatmap(ax_inactive, inactive)

	plt.show()


### GRAPHICS:
# MEAN AND STD DEV FOR 5X5, 10000 TRIALS
# MOST COMMON 'LONGEST WORD' FOR 5X5, 10000 TRIALS
# AVG POINTS AVAILABLE AGAINST BOARD SIZE

if __name__ == "__main__":
	point_distribution()