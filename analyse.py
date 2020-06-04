"""Functions for performing analysis on boggle boards"""

from boggle import Boggle
from solver import Solver
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm, colors
from tqdm import tqdm
import multiprocessing as mp

def run(n=1, board_id=None, processors=5, chunk_size=1000, batch_size = 5000):
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

	solver = Solver() # Load solver

	pool = mp.Pool(processors)

	# produce array of batch sizes
	batches = [batch_size] * (n//batch_size)
	if n%batch_size>0: batches += [n % batch_size]

	layouts = []
	solutions = []
	with tqdm(total=n) as progress:
		for batch_size in batches:
			batch_layouts = [board.gen() for i in range(batch_size)]
			batch_sols = pool.map(solver.solve, batch_layouts, chunksize=chunk_size)

			layouts += batch_layouts
			solutions += batch_sols

			progress.update(batch_size)

	return layouts, solutions


def board_heatmap(ax, data, cmap="jet"):
	norm = colors.Normalize(vmin=0, vmax=data.max())
	ax.imshow(data, cmap=cmap, norm=norm)
	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	plt.colorbar(sm, ax = ax)

def point_distribution():
	"""Run experiment to measure distribution of points across board"""

	n = 50000
	layouts, solutions = run(n=n, processors=n_cpu)

	# calculate point distribution throughout board
	points = np.zeros((n, 5, 5))
	for i, sol in enumerate(solutions):
		for word in sol:
			val = len(word) - 3
			all_coords = set([coord for route in sol[word] for coord in route]) # all UNIQUE coords in routes
			for coord in all_coords:
				x, y = coord
				points[i, y, x] += val

	means = points.mean(axis=0)
	vars = np.std(points, axis=0)
	inactive = np.mean(points == 0, axis=0)

	cmap = "jet"

	fig, (ax_means, ax_std, ax_inactive) = plt.subplots(ncols=3)

	## plot means
	board_heatmap(ax_means, means)
	board_heatmap(ax_std, vars)
	board_heatmap(ax_inactive, inactive)

	plt.show()


def word_distribution():
	"""Run experiment to measure the distribution of high scoring/populous words across games"""

	n = 250000
	layouts, points, solutions = run(n=n)

	### Make dict of each word : num appearances (repeats ignored)
	appearances = {} # All valid words : num appearances (repeats ignored)
	apperances_longest = {} # word : num appearances as longest word(s) in game (repeats ignored)
	for i, sol in enumerate(solutions):
		if len(sol) == 0: continue
		max_length = max([len(word) for word in sol]) # length of longest word(s)
		for word in sol:
			add_to = [appearances] # dictionaries to add to
			if len(word) == max_length:
				add_to += [apperances_longest]
			for record in add_to:
				record[word] = record.get(word, 0) + 1

	ranked_by_appearance = sorted(appearances.keys(), key=lambda x: appearances[x])
	ranked_by_appearance_longest = sorted(apperances_longest.keys(), key=lambda x: apperances_longest[x])

	# print(apperances_longest)

	print({k: apperances_longest[k] for k in ranked_by_appearance_longest[-10:]})

### GRAPHICS:
# MEAN AND STD DEV FOR 5X5, 10000 TRIALS
# MOST COMMON 'LONGEST WORD' FOR 5X5, 10000 TRIALS
# AVG POINTS AVAILABLE AGAINST BOARD SIZE

if __name__ == "__main__":
	n_cpu = mp.cpu_count()
	print("Number of processors: ", n_cpu)

	point_distribution()
	# word_distribution()