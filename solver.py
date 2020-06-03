"""Various methods that all aim to solve a boggle board.

Inputs:
list of strings. List length, and string length are equal."""

import json, os
from typing import List

class Node:
	def __init__(self, value):
		self.v = value
		self.children: List['Node'] = []

	def child_with_value(self, letter) -> 'Node':
		"""Returns the child with value `letter`. If no such child exists, make a new one"""
		for child in self.children:
			if child.v == letter:
				return child
		# if none exists:
		new_child = Node(letter)
		self.add_child(new_child)
		return new_child


	def add_child(self, other: 'Node'):
		self.children.append(other)

	def __repr__(self):
		return f"{self.v}:{self.children}"


class Dictionary:

	def __init__(self, src="CSW19"):

		with open(os.path.join("dict", src+".json")) as infile:
			self.lookup:dict = json.load(infile)
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
				prev_node = prev_node.child_with_value(c)

		return tree_head






class Solver:
	def __init__(self, board):
		pass

if __name__ == "__main__":
	d = Dictionary()
	print(d.tree)
