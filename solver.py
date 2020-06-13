"""Various methods that all aim to solve a boggle board.

Inputs:
list of strings. List length, and string length are equal."""


from itertools import product
from time import perf_counter

from boggle import Boggle
from dict import Dictionary
import numpy as np

from utils import Timer, timeit
timer = Timer()

possible_moves = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1,-1], [1,0], [1,1]]

clip = lambda X, width=5: np.clip(X, a_min=0, a_max = width-1)


def gen_adjacents(width):
	"""Given a board size <width>, generates a dict in which key (y, x) gives the list of valid (y, x) neighbour coordinates
	not including itself

	eg arr[0,0], width=5 = [(1,0), (0,1), (1,1)]
	"""
	adjacents = {} # np.zeros((width, width, width, width), dtype=np.bool)
	for y in range(width):
		for x in range(width):
			y_start, y_end = sorted([0, y-1, y+2, width])[1:3]
			x_start, x_end = sorted([0, x-1, x+2, width])[1:3]
			# all valid coordinates, non-inclusive of x,y
			adjacents[(y, x)] = [(j, i) for j in range(y_start, y_end) for i in range(x_start, x_end) if (i,j)!=(x,y)]

	return adjacents

def get_next_moves(adjacents, unused_tiles):
	"""Yields coordinates of possible next letters
	Does not allow for coordinates outside of the board, or repeated use of the same tile"""

	valid_moves = filter(lambda X: unused_tiles[X], adjacents)
	return valid_moves

def point_encoder(point, width=5):
	"""Encodes coord as integer"""
	x, y = point
	return y*width + x

def point_decoder(val, width=5):
	"""Extracts 2d coord from given integer"""
	x, y = val % width, val//width
	return (x,y)

class Solver:
	def __init__(self, dict_src = "CSW19", width=5):
		self.d = Dictionary(dict_src)

		self.width = width

		# numpy array of size (width)^4, which gives the adjacents mask for a given x,y. Generated initially to avoid repeats
		self.adjacents = gen_adjacents(width)

	def solve(self, board):
		assert len(board)==self.width, f"Board must be width {self.width} - got width {len(board)}"
		width = self.width
		starts = product(range(width), repeat=2)

		tree_head = self.d.tree
		# starts contains the coordinates for each tile in the grid
		all_words = {}

		# memory for storing unused tiles. Each word to be tried is given an index associated with this memory
		unused_tiles = np.ones((1000, width, width), dtype=np.bool)

		for x, y in starts:
			c = board[y][x]
			if c == "Q": c = "QU"
			prev_node = tree_head[c] # guaranteed to already exist

			unused_tiles[:] = True # rest unused tiles memory
			unused_tiles_counter = 0
			unused_tiles[:, y, x] = 0

			# each word is stored as [word_string, current_coords, idx_of_used_tiles, prev_node]
			first_word = [c, (y, x), unused_tiles_counter, prev_node]
			cur_words = [first_word]

			# keep looping until no new words can be made
			while cur_words:
				prev_words = cur_words
				cur_words = []

				for word in prev_words:
					word_string, (word_y, word_x), unused_tiles_idx, prev_node = word

					next_moves = get_next_moves(self.adjacents[(word_y, word_x)], unused_tiles[unused_tiles_idx])

					for j, i in next_moves:
						new_c = board[j][i]
						if new_c == "Q":
							new_c = "QU"
						next_node = prev_node.get(new_c)

						if next_node is not None:
							unused_tiles_counter += 1
							unused_tiles[unused_tiles_counter] = unused_tiles[unused_tiles_idx] # copy prev used tiles array
							unused_tiles[unused_tiles_counter, j, i] = 0 # set this tile to used

							new_word = [word_string + new_c, (j, i), unused_tiles_counter, next_node]
							cur_words.append(new_word)
							# check if this is a real word
							# => has no children, or is in the dictionary
							if not next_node or new_word[0] in self.d:
								new_string = new_word[0]
								new_path = new_word[2]
								# all_words stores the word, and all paths to make that word
								all_words[new_string] = all_words.get(new_string, []) + [new_path]

		return all_words

if __name__ == "__main__":
	board = Boggle()
	s = Solver()

	N = 300
	t0 = perf_counter()
	ts = []

	for i in range(N):
		s.solve(board.gen())
		ts.append(perf_counter() - t0)
		t0 = perf_counter()

	print(f"Time to solve: {sum(ts)/len(ts)*1000:.1f}ms")

	# SOLVE BEST - EXPECTED ANSWER - 2945
	t0 = perf_counter()
	sol = s.solve(["EASAN", "BLRME", "AUIEE", "STSNS", "NURIG"])
	print(sum(len(w)-3 for w in sol), f"[{(perf_counter()-t0) * 1000:.1f}ms]")
	by_length = {}
	for w in sol:
		by_length[len(w)] = by_length.get(len(w), []) + [w]
	print(", ".join([f"{i}: {len(v)}" for i, v in by_length.items()]))
	print(by_length[11])

	print(timer.report(nits=N, make_dict=1))