"""Boggle class, defines boggle board, and can find all valid solutions"""

import json, os
import numpy as np
import random

class Boggle:

    def __init__(self, board="2003super"):

        with open(os.path.join("boards", board+".json")) as infile:
            self.block_data = json.load(infile)

        # Convert to list of size nblocks (with repeats)
        self.blocks = []
        for string, count in self.block_data.items():
            self.blocks += [string] * count

        nblocks = len(self.blocks)

        self.width = width = int(nblocks ** .5)

        assert nblocks == width ** 2, f"nblocks = {nblocks} is not a square number. {board}"

    def gen(self, return_dice = False):
        """Generate a width x width valid arrangement.
        return dice flag used to return the order of dice accessed, for optimisation"""

        out = [""] * self.width

        np.random.shuffle(self.blocks) # shuffle blocks

        for n in range(self.width**2):
            row, col = n//self.width, n%self.width
            letter = random.choice(self.blocks[n])

            out[row] += letter

        if return_dice:
            return out, self.blocks.copy()
        else:
            return out



if __name__ == "__main__":
    b = Boggle()
    print(b.gen(return_dice=True))