"""Functions for performing analysis on boggle boards"""

from boggle import Boggle
from solver import Solver
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

from tqdm import tqdm
import multiprocessing as mp
import os, json

def run(n=1, board_id=None, processors=5, chunk_size=2000, batch_size = 10000, outname="2003super5x5"):
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

	max_words_per_board = 1200 # store maximum number of valid entries per board

	solver = Solver() # Load solver
	max_word_length = solver.d.max_length

	pool = mp.Pool(processors)

	# produce array of batch sizes
	batches = [batch_size] * (n//batch_size)
	if n%batch_size>0: batches += [n % batch_size]

	# initialise data storage arrays
	layouts = np.empty((n, board.width, board.width), dtype="<U1") # list of each board
	solutions = np.empty((n, max_words_per_board), dtype = f"|S{max_word_length}") # list of valid words for each board
	points = np.zeros((n, 5, 5), dtype=np.uint64) # point distribution across each board

	with tqdm(total=n) as progress:
		for b, batch_size in enumerate(batches):
			batch_layouts = [board.gen() for i in range(batch_size)]
			batch_sols = pool.map(solver.solve, batch_layouts, chunksize=chunk_size)

			# compute and save data
			for i_inbatch, sol in enumerate(batch_sols):
				i = b*batch_size + i_inbatch # overall idx
				for word in sol:
					val = len(word) - 3
					all_coords = set(
						[tuple(coord) for route in sol[word] for coord in route])  # all UNIQUE coords in routes
					for coord in all_coords:
						x, y = coord
						points[i, y, x] += val

				layouts[i] = [list(r) for r in batch_layouts[i_inbatch]]
				solutions[i][:len(sol.keys())] = list(sol.keys())

			progress.update(batch_size)


	out = dict(layouts = layouts,
						solutions = solutions,
						points = points)

	np.savez_compressed(os.path.join("results", outname),
						width = 5, height = 5, **out
						)

def load_run(run, load_keys = ["layouts", "solutions", "points", "width", "height"]):
	res = {}
	with np.load(os.path.join("results", run+".npz")) as data:
		for k in load_keys:
			res[k] = data[k]

	print(f"{run} loaded.")

	return res

def board_heatmap(ax, data, cmap="jet", title="", font = "Arial", cbar_fmt=None):
	norm = colors.Normalize(vmin=0, vmax=data.max())
	ax.imshow(data, cmap=cmap, norm=norm)

	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)
	sm = cm.ScalarMappable(cmap=cmap, norm=norm)
	if cbar_fmt is None:	plt.colorbar(sm, cax=cax)
	elif cbar_fmt == "pct": plt.colorbar(sm, cax=cax, format=FuncFormatter(lambda x, pos: f"{100*x:.1f}%"))
	else: raise ValueError(f"cbar_fmt '{cbar_fmt}' not recognised.")

	if title != None:
		ax.set_title(title, fontname = font, fontweight="bold")

def point_distribution(run="2003super5x5"):
	"""Run experiment to measure distribution of points across board"""

	res = load_run(run, load_keys=["points"])

	points = res['points']
	n = len(points)
	means = points.mean(axis=0)
	vars = np.std(points, axis=0)
	inactive = np.mean(points == 0, axis=0)

	cmap = "Oranges"

	fig, (ax_means, ax_std, ax_inactive) = plt.subplots(ncols=3, figsize= (8, 2.5))

	[ax.axis("off") for ax in (ax_means, ax_std, ax_inactive)]

	## plot means
	board_heatmap(ax_means, means, cmap=cmap, title="Mean score")
	board_heatmap(ax_std, vars, cmap=cmap, title="Standard dev")
	board_heatmap(ax_inactive, inactive, cmap=cmap, title="Probability of\n0 points", cbar_fmt="pct")

	plt.subplots_adjust(wspace=0.5, left=.02, right=.93, top=1, bottom=0)
	plt.show()


def word_distribution(run="2003super5x5"):
	"""Run experiment to measure the distribution of high scoring/populous words across games"""
	res = load_run(run, load_keys=["solutions"])
	solutions = res['solutions']

	n = len(solutions)

	### Make dict of each word : num appearances (repeats ignored)
	appearances = {}  # All valid words : num appearances (repeats ignored)
	apperances_longest = {}  # word : num appearances as longest word(s) in game (repeats ignored)

	# This method requires batching, as insufficient memory for large datasets
	batch_size = min(1000, n//10)
	n_batches = n // batch_size
	batches = np.array_split(solutions, n_batches)

	with tqdm(total=n) as progress_bar:
		for batch in batches:
			# fast method for getting lengths of whole array. src: https://stackoverflow.com/questions/44587746/length-of-each-string-in-a-numpy-array
			A = batch.astype(np.str)
			v = A.view(np.uint32).reshape(A.size, -1)
			l = np.argmin(v, 1)
			l[v[np.arange(len(v)), l] > 0] = v.shape[-1]
			l = l.reshape(A.shape)

			longest_lengths = np.max(l, axis=1) # longest word per board
			longest_words_idxs = np.argwhere(l == longest_lengths[:, None]) # (board, pos) for each longest word occurance

			for board, pos in longest_words_idxs:
				word = batch[board, pos]
				apperances_longest[word] = apperances_longest.get(word, 0) + 1

			progress_bar.update(len(batch))

	# ranked_by_appearance = sorted(appearances.keys(), key=lambda x: appearances[x])
	ranked_by_appearance_longest = sorted(apperances_longest.keys(), key=lambda x: apperances_longest[x])

	# print({k: apperances_longest[k] for k in ranked_by_appearance[-10:]})
	print("10 most common: ", {k: apperances_longest[k] for k in ranked_by_appearance_longest[-11:]})
	print("10 least common: ", {k: apperances_longest[k] for k in ranked_by_appearance_longest[:10]})

def inactive_distribution(run="2003super5x5"):
	"""Run experiment to measure the distribution of inactive tiles (tiles with no valid words connected)"""

	res = load_run(run, load_keys=["points"])

	points = res['points']
	n = len(points)

	print((points.sum(axis=(1,2))==0).sum())

	inactive_by_game = np.count_nonzero(points==0, axis=(1,2)) # list of number of inactive tiles for each game

	ninactive = np.arange(0, 26, 1)
	count_by_ninactive = np.array([(inactive_by_game==i).sum() for i in ninactive]) # list of number of games with exactly <idx> inactive tiles

	plt.bar(ninactive, count_by_ninactive / n)
	print(" ".join([f"({j/1e6}, {i})[{j}]" for i, j in zip(ninactive, count_by_ninactive)]))
	plt.show()

### GRAPHICS:
# MEAN AND STD DEV FOR 5X5, 10000 TRIALS
# MOST COMMON 'LONGEST WORD' FOR 5X5, 10000 TRIALS
# AVG POINTS AVAILABLE AGAINST BOARD SIZE
# % distribution of number of inactive tiles

if __name__ == "__main__":

	## RUN
	# n_cpu = mp.cpu_count()
	# print("Number of processors: ", n_cpu)
	# run(n=2000, processors=n_cpu, outname="2003super5x5_small")

	point_distribution()
	# word_distribution(run="2003super5x5")
	# inactive_distribution()