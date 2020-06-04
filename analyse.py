"""Functions for performing analysis on boggle boards"""

from boggle import Boggle
from solver import Solver
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
import multiprocessing as mp
import os, json

def run(n=1, board_id=None, processors=5, chunk_size=1000, batch_size = 5000, outname="2003super5x5"):
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

	out = {
		"width":5,
		"height":5,
		"layouts":layouts,
		"solutions": solutions,
	}

	with open(os.path.join("results", outname+".json"), "w") as outfile:
		json.dump(out, outfile)

def load_run(run):
	with open(os.path.join("results", run+".json")) as infile:
		res = json.load(infile)
	return res

def board_heatmap(ax, data, cmap="jet", title=""):
	norm = colors.Normalize(vmin=0, vmax=data.max())
	ax.imshow(data, cmap=cmap, norm=norm)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	plt.colorbar(sm, cax=cax)

	if title != None:
		ax.set_title(title)

def point_distribution(run="2003super5x5"):
	"""Run experiment to measure distribution of points across board"""

	res = load_run(run)
	layouts, solutions = res['layouts'], res['solutions']
	n = len(layouts)

	# calculate point distribution throughout board
	points = np.zeros((n, 5, 5))
	for i, sol in enumerate(tqdm(solutions)):
		for word in sol:
			val = len(word) - 3
			all_coords = set([tuple(coord) for route in sol[word] for coord in route]) # all UNIQUE coords in routes
			for coord in all_coords:
				x, y = coord
				points[i, y, x] += val

	means = points.mean(axis=0)
	vars = np.std(points, axis=0)
	inactive = np.mean(points == 0, axis=0)

	cmap = "Reds"

	fig, (ax_means, ax_std, ax_inactive) = plt.subplots(ncols=3)

	## plot means
	board_heatmap(ax_means, means, cmap=cmap, title="Mean score")
	board_heatmap(ax_std, vars, cmap=cmap, title="Standard dev")
	board_heatmap(ax_inactive, inactive, cmap=cmap, title="Probability of 0 points")

	plt.subplots_adjust(wspace=0.5)
	plt.show()


def word_distribution(run="2003super5x5"):
	"""Run experiment to measure the distribution of high scoring/populous words across games"""

	res = load_run(run)
	layouts, solutions = res['layouts'], res['solutions']
	n = len(layouts)

	print("HERE")

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

	print({k: apperances_longest[k] for k in ranked_by_appearance[-10:]})
	print({k: apperances_longest[k] for k in ranked_by_appearance_longest[-10:]})

def inactive_distribution(run="2003super5x5"):
	"""Run experiment to measure the distribution of inactive tiles (tiles with no valid words connected)"""

	res = load_run(run)
	layouts, solutions = res['layouts'], res['solutions']
	n = len(layouts)

	# calculate point distribution throughout board
	points = np.zeros((n, 5, 5))
	for i, sol in enumerate(tqdm(solutions)):
		for word in sol:
			val = len(word) - 3
			all_coords = set([tuple(coord) for route in sol[word] for coord in route]) # all UNIQUE coords in routes
			for coord in all_coords:
				x, y = coord
				points[i, y, x] += val

	inactive_by_game = np.count_nonzero(points==0, axis=(1,2)) # list of number of inactive tiles for each game

	ninactive = np.arange(0, 26, 1)
	count_by_ninactive = np.array([(inactive_by_game==i).sum() for i in ninactive]) # list of number of games with exactly <idx> inactive tiles

	plt.bar(ninactive, count_by_ninactive / n)
	plt.show()


### GRAPHICS:
# MEAN AND STD DEV FOR 5X5, 10000 TRIALS
# MOST COMMON 'LONGEST WORD' FOR 5X5, 10000 TRIALS
# AVG POINTS AVAILABLE AGAINST BOARD SIZE
# % distribution of number of inactive tiles

if __name__ == "__main__":

	## RUN
	n_cpu = mp.cpu_count()
	print("Number of processors: ", n_cpu)
	run(n=25000, processors=n_cpu)

	# point_distribution()
	# word_distribution()
	# inactive_distribution()