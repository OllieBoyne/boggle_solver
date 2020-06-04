"""Various methods that all aim to solve a boggle board.

Inputs:
list of strings. List length, and string length are equal."""


from itertools import product
from time import perf_counter

from boggle import Boggle
from dict import Dictionary


def get_next_moves(x, y, width, used_tiles):
	"""Yields coordinates of possible next letters
	Does not allow for coordinates outside of the board, or repeated use of the same tile"""
	for dx, dy in product(range(-1, 2), repeat=2):
		X, Y = x + dx, y + dy
		if 0 <= X < width and 0 <= Y < width and (X, Y) not in used_tiles:
			yield X, Y


class Solver:
	def __init__(self, dict_src = "CSW19"):
		self.d = Dictionary(dict_src)

	def solve(self, board):
		width = len(board)
		starts = product(range(width), repeat=2)

		tree_head = self.d.tree
		# starts contains the coordinates for each tile in the grid
		all_words = {}
		for x, y in starts:
			c = board[y][x]
			if c == "Q": c = "QU"
			prev_node = tree_head[c] # guaranteed to already exist


			# each word is stored as [word_string, current_coords, [used_tiles], prev_node]
			first_word = [c, (x, y), [(x, y)], prev_node]
			cur_words = [first_word]

			# keep looping until no new words can be made
			while cur_words:
				prev_words = cur_words
				cur_words = []
				for word in prev_words:
					word_string = word[0]
					word_x, word_y = word[1]
					used_tiles = word[2]
					prev_node = word[3]
					for i, j in get_next_moves(word_x, word_y, width, used_tiles):
						new_c = board[j][i]
						if new_c == "Q":
							new_c = "QU"
						next_node = prev_node.get(new_c)

						if next_node is not None:
							new_word = [word_string + new_c, (i, j), used_tiles + [(i, j)], next_node]
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
	board = Boggle().gen()
	s = Solver()
	t1 = perf_counter()
	print(s.solve(board))
	print(f"Time to solve: {perf_counter() - t1}")