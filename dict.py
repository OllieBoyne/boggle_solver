
import json, os
from typing import List
import re
from time import perf_counter

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
		tree_head = {} # top of tree is an empty string

		for word in self.lookup.keys():
			prev_node = tree_head
			for i, c in enumerate(word):
				# if previous letter was a q, then the u has already been added
				if word[i-1] == "Q":
					continue
				if c == "Q":
					c = "QU"

				new_node = prev_node.get(c, None)
				if new_node is None:
					prev_node[c] = {}
					new_node = prev_node[c]
				prev_node = new_node

		return tree_head

def process_dict(src = r"CSW19"):
	"""Given a list of .txt files of all words in a dictionary, applies some filters and saves as a JSON dict of
	word : length.

	Filters:
	- discard < 3 letters
	- discard any word with an instance of Q not followed by a U (not valid in Boggle)"""

	with open(os.path.join("dict", src+".txt")) as infile:
		words = [l.rstrip() for l in infile.readlines()] # list of all words in dict


	length_filter = lambda s: len(s) < 4

	q_regex = re.compile(r"q([^u]|$)",re.IGNORECASE) # Q followed by NOT U (any case)
	q_filter = lambda s: bool(q_regex.search(s))

	accepted = lambda s: not any([length_filter(s), q_filter(s)])

	filt_words = list(filter(accepted, words)) # list of filtered words

	out = {word : len(word) for word in filt_words}

	with open(os.path.join("dict", src+".json"), "w") as outfile:
		json.dump(out, outfile)

if __name__ == "__main__":
	# process_dict()
	d = Dictionary()
	print(d.tree)