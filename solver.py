"""Various methods that all aim to solve a boggle board.

Inputs:
list of strings. List length, and string length are equal."""

import json, os
from typing import List
from itertools import product
from time import perf_counter

from boggle import Boggle


def get_next_moves(x, y, width, used_tiles):
	"""Yields coordinates of possible next letters
	Does not allow for coordinates outside of the board, or repeated use of the same tile"""
	for dx, dy in product(range(-1, 2), repeat=2):
		X, Y = x + dx, y + dy
		if 0 <= X < width and 0 <= Y < width and (X, Y) not in used_tiles:
			yield X, Y



class Node:
	def __init__(self, value):
		self.v = value
		self.children: List['Node'] = []

	def child_with_value(self, letter):
		"""Returns the child with value `letter`."""
		for child in self.children:
			if child.v == letter:
				return child
		# if none exists:
		return None


	def add_child(self, other: 'Node'):
		self.children.append(other)

	def __repr__(self):
		return self.v


class Dictionary:

	def __init__(self, src="CSW19"):

		with open(os.path.join("dict", src+".json")) as infile:
			self.lookup = json.load(infile)
		self.size = len(self.lookup)
		self.tree = self.make_tree()

	def __len__(self):
		return self.size

	def __contains__(self, word):
		return word in self.lookup

	def make_tree(self):
		"""Creates a tree that contains all words in self.lookup"""
		longest_word = max(self.lookup.values())
		tree_head = Node('') # top of tree is an empty string

		for word in self.lookup.keys():
			prev_node = tree_head
			for c in word:
				new_node = prev_node.child_with_value(c)
				if new_node is None:
					new_node = Node(c)
					prev_node.add_child(new_node)
				prev_node = new_node

		return tree_head


class Solver:
	def __init__(self, dictionary, board):
		self.d:Dictionary = dictionary
		self.board = board

	def solve(self):
		width = len(self.board)
		starts = product(range(width), repeat=2)

		tree_head = self.d.tree
		# starts contains the coordinates for each tile in the grid
		all_words = {}
		for x, y in starts:
			c = self.board[y][x]
			prev_node = tree_head.child_with_value(c) # guaranteed to already exist
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
						new_c = self.board[j][i]
						next_node = prev_node.child_with_value(new_c)

						if next_node is not None:
							new_word = [word_string + new_c, (i, j), used_tiles + [(i, j)], next_node]
							cur_words.append(new_word)

							# check if this is a real word
							# => has no children, or is in the dictionary
							if not next_node.children or new_word[0] in self.d:
								new_string = new_word[0]
								new_path = new_word[2]
								# all_words stores the word, and all paths to make that word
								all_words[new_string] = all_words.get(new_string, []) + [new_path]

		print(all_words)
		return all_words



if __name__ == "__main__":
	board = Boggle().gen()
	t0 = perf_counter()
	d = Dictionary()
	print(f"Time to make tree: {perf_counter() - t0}")

	s = Solver(d, board)
	t1 = perf_counter()
	s.solve()
	print(f"Time to solve: {perf_counter() - t1}")