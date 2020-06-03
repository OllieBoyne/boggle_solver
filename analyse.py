"""Functions for performing analysis on boggle boards"""

from boggle import Boggle
from solver import solve

def run(n=10, board_id=None):
	"""
	Solves several generated boggle configs

	:param method: solving method
	:param n: number of boards to run
	:param board: name of board to use
	:return: n board layouts, n board solutions
	"""

	if board_id is None:
		board = Boggle()
	else:
		board = Boggle(board=board_id)

	layouts = []
	solutions = []

	for i in range(n):
		lay = board.gen()
		layouts.append(lay)
		sol = solve(lay)
		solutions.append(sol)

	return layouts, solutions


if __name__ == "__main__":
	run()