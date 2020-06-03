"""Various methods that all aim to solve a boggle board.

Inputs:
list of strings. List length, and string length are equal."""

import json, os

class Dictionary:

    def __init__(self, src="CSW19"):

        with os.path.join("dict", src+".json") as infile:
            self.lookup = json.load(infile)
        self.size = len(self.lookup)

    def __len__(self):
        return self.size

    def __contains__(self, word):
        return word in self.lookup

def brute_force():
    pass