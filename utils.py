from time import perf_counter, time, perf_counter_ns, sleep, process_time, monotonic, thread_time
import numpy as np

tfunc = time # fastest timing function currently

class Timer:
	def __init__(self):
		self.t0 = tfunc()
		self.log = {}

	def add(self, label):
		"""nrepeats: optional value to multiply each value by.
		Either int, or iterable with valid length
		Used for timing the total time for an entire loop -
		nrepeats is length of iterator."""

		if label not in self.log: self.log[label] = []
		self.log[label] += [tfunc() - self.t0]
		self.t0 = tfunc()

	def report(self, nits=None, **custom_nits):
		"""Print report of log.
		if nits is none, assume the mean time for each operation is required.
		if nits is an int, divide the total time by nits
		any nits that differ can be given in custom_nits"""

		out = {}
		for k, t in self.log.items():
			if nits is None:
				out_time = np.mean(t)
			elif isinstance(nits, int):
				if k in custom_nits:
					out_time = np.sum(t) / custom_nits[k]
				else:
					out_time = np.sum(t) / nits

			out[k] = out_time

		return "\n".join([f"{k} = {t*1000:.1f}ms" for k, t in out.items()])

def timeit(func, *args, **kwargs):
	"""Time the operation of a given function"""
	t0 = perf_counter()
	N = 10000
	for n in range(N):
		func(*args, **kwargs)
	print(f"{func.__name__}, {(perf_counter()-t0)*1000000/N:.3f}us")

if __name__ == "__main__":
	time_funcs = [time, perf_counter, perf_counter_ns, process_time, monotonic, thread_time]
	for f in time_funcs:
		timeit(f)
