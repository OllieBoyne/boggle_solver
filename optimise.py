"""Algorithm to find the best possible boggle board.

The algortihm is stochastic, and generates N random boards to be optimised, and makes modifications to each to approach and optimal solution.

Individual dice are randomly 'rerolled' with a probability Ae^-p, where p is proportional to the point value of that block.
Pairs of dice are randomly 'switched' with a probability Be^-(p+q)/2, where p and q are proportional to the point values of the two blocks.
Boards that have not improved for a certain period of time are returned to their current best.
Boards that are performing poorly (very low best score for many iterations) are completely regenerated.

"""

from solver import Solver
from boggle import Boggle
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

outloc = "outputs/best.txt"

# Hyper params
N = 20
n_it = 3
board_size = 5
log_window = 10 # moving average window for log plot

REGEN_NORM = 100 # Divide its since best by this to get fitness indicator
RETURN_TO_BEST_AFTER = 50 # if this board hasn't improved for a certain amount of time, return to best for this board

# reroll dice with prob REROLL_COEFF * exp(- points / board_max)
REROLL_COEFF = 0.08
MAX_REROLLS = 5 # max rerolls in single it

# swap dice with prob SWAP_COEFF * exp(- p * q / (board max**2) )
SWAP_COEFF = 0.001
MAX_SWAPS = 5 # max swaps in single it

solver = Solver()
boggle = Boggle()

from utils import Timer

board_timer = Timer()

def layout_to_str(arr):
	"""Convert array of size (board_size, board_size) to correct format string for solver"""
	return ["".join(row) for row in arr]

def running_mean(x, N):
	cumsum = np.cumsum(np.insert(x, 0, 0))
	return (cumsum[N:] - cumsum[:-N]) / float(N)

def weighted_selection(probs):
	"""Given an array of probablities M, selects a random one, weighted to the array. Returns the index of the selected one"""
	assert abs(probs.sum()-1) <= 1e-3, f"Sum of probabilities must equal 1. Sum = {probs.sum():.5f}"
	p = np.random.randn()
	c = 0
	for n, w in enumerate(probs):
		c += w
		if p >= c: return n
	return n

class Board:
	"""Object that is continually optimised throughout process. Stores board configuration, as well as point distribution"""

	def __init__(self):
		self.points = np.zeros((board_size, board_size))
		self.layout = None
		self.score = None

		layout, dice = boggle.gen(return_dice=True)

		self.layout = np.array([list(r) for r in layout], dtype="<U1").reshape(board_size, board_size)
		self.dice = np.array([list(d) for d in dice], dtype="<U1").reshape(board_size, board_size, 6) # (width x width x 6 array of dice)

		self.best_score = 0
		self.best_layout = layout
		self.best_dice = dice
		self.it_since_best = 0 # iterations since previous best
		self.eval_board()

	def eval_board(self, increment=True):
		board_timer.add("_")
		layout = self.layout
		self.score = 0
		layout_str = layout_to_str(layout)
		board_timer.add("layout to str")
		sol = solver.solve(layout_str)
		board_timer.add("solve")

		# compute point distribution
		for word in sol:
			val = len(word) - 3
			self.score += val
			all_coords = set(
				[tuple(coord) for route in sol[word] for coord in route])  # all UNIQUE coords in routes
			for coord in all_coords:
				x, y = coord
				self.points[y, x] += val

		board_timer.add("points")

		if self.score >= self.best_score:
			self.best_score = self.score
			self.best_layout = layout.copy()
			self.best_dice = self.dice.copy()
			self.it_since_best = 0  # reset counter

		elif increment:
			self.it_since_best += 1

		board_timer.add("eval best")

		return self.score

	def return_to_best(self):
		"""Return to previous best layout"""
		self.layout = self.best_layout
		self.dice = self.best_dice

	def save(self):
		with open(outloc, "w") as outfile:
			outfile.write("\n".join([f"SCORE: {self.best_score}"] + layout_to_str(self.best_layout)))

	def __gt__(self, other):
		return self.score > other.score

	def reroll(self, x, y):
		"""Reroll the (x,y)th die"""

		cur = self.layout[y][x]
		die = self.dice[y][x]

		letters_to_choose = die[die != cur]
		score_by_letter = np.zeros(letters_to_choose.shape)
		for n, l in enumerate(letters_to_choose):
			self.layout[y][x] = l
			score_by_letter[n] = self.eval_board(increment=False)

		prob_select = np.log(score_by_letter/(score_by_letter.min() + 1e-3))
		prob_select = prob_select / prob_select.sum() # normalise to get 0 < p <= 1
		selection = letters_to_choose[weighted_selection(prob_select)]

		### this has introduced some errors

		self.layout[y][x] = selection # assign to board

	def swap(self, x1, y1, x2, y2):
		"""Swaps the (x1, y1)th die with the (x2, y2)th die, and changes the layout accordingly"""

		self.dice[y1,x1], self.dice[y2, x2] = self.dice[y2, x2], self.dice[y1, x1] # switch die
		self.layout[y1,x1], self.layout[y2, x2] = self.layout[y2, x2], self.layout[y1, x1] # switch layout

	def copy(self):
		"""Return copy of board object for storing best."""

		b = Board()
		b.layout = self.layout
		b.dice = self.dice
		b.eval_board()
		return b



def run():

	log = {} # log of best score

	boards = [Board() for _ in range(N)] # change to N times
	best = boards[0].copy() # best board

	timer = Timer()

	iterations = np.arange(n_it)
	with tqdm(iterations) as tqdm_iterator:
		for n in tqdm_iterator:
			stats = {"rerolls":0, "swaps":0, "regens":0, "returns":0} # track number of swaps
			scores = [b.best_score for b in boards]
			cur_mean = np.mean(scores) # track current mean best for evaluating fitness
			upper_cutoff = np.percentile(scores, 90)

			timer.add("stats")

			for board in boards:
				if board > best:
					best = board.copy()

				if board.points.max() > 0: # if score is 0, will be regened later on with 100% probability
					## make modifications
					prob_reroll = REROLL_COEFF * np.exp(-board.points/board.points.max())
					to_reroll = np.where(np.random.rand(board_size, board_size) < prob_reroll)
					timer.add("prob reroll")

					for y, x in zip(*to_reroll):
						stats['rerolls'] += 1
						board.reroll(x, y)

					timer.add("reroll")

					swap_corr = np.einsum("ij,kl->ijkl", board.points, board.points)
					prob_swap = SWAP_COEFF * np.exp(-swap_corr / (board.points.max()**2))

					to_swap = np.where(np.random.rand(*[board_size]*4) < prob_swap)
					timer.add("prob swap")

					for y1, x1, y2, x2 in zip(*to_swap):
						stats['swaps'] += 1
						board.swap(x1, y1, x2, y2)

					timer.add("swap")
					board.eval_board()
					timer.add("eval board")

				# Regenerate poorly performing boards
				# any board in top 10% of performers is automatically not rejected
				# boards with lower than mean score have increased chance of being regenerated
				# the rate of regeneration increases with iterations
				prob_regen = (board.best_score < upper_cutoff) * np.exp(-(board.best_score / cur_mean) * (1/(board.it_since_best+1)))

				if np.random.rand() < prob_regen:
					stats['regens'] += 1
					board.__init__()

				# Return any boards to their best that have not improved in smaller number of iterations
				if board.it_since_best % RETURN_TO_BEST_AFTER == 0:
					stats['returns'] += 1
					board.return_to_best()

				timer.add("regens & returns")

			scores = [b.best_score for b in boards]
			board_stats = f"MAX BEST = {best.score:04d}, MIN BEST = {min(scores):04d}, MEAN BEST = {int(cur_mean):04d}, 10P = {int(upper_cutoff):04d}, "
			stats_string = ", ".join([f"{k}:{n/N:.3f}" for k, n in stats.items()])
			tqdm_iterator.set_description(board_stats + stats_string)

			log['best'] = log.get("best", []) + [best.score]
			log['mean_best'] = log.get("mean_best", []) + [cur_mean]
			log['upper_cutoff'] = log.get("upper_cutoff", []) + [upper_cutoff]


	print(timer)
	print("EVAL TIMER:")
	print(board_timer)

	best.save()

	fig, ax = plt.subplots()
	for k, data in log.items():
		ax.plot(data, label=k)
		mov_av = running_mean(data, log_window)
		mov_av = np.pad(mov_av, pad_width=((log_window//2, log_window//2)), constant_values=np.nan)
		ax.plot(mov_av, "--", label=k+"_av")

	ax.legend()
	plt.show()

if __name__ == "__main__":
	run()
