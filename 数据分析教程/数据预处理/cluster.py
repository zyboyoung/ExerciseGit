import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_blobs, make_moons
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from functools import wraps
import time


def caltime(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print('-' * 8)
		print('start time: ', time.asctime(time.localtime(start)))
		print('end time:   ', time.asctime(time.localtime(end)))
		print('-' * 8)
		cost_time = end - start
		if cost_time < 1:
			print(func.__name__, '{:.5f}'.format(cost_time * 1000), 'ms')
		else:
			cost_time = '{:.2f}'.format(cost_time)
			print(func.__name__, cost_time, 's')
		print('-' * 8)
		return result

	return wrapper


@caltime
def main():
	n_samples = 1000
	circles = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
	moons = make_moons(n_samples=n_samples, noise=0.05)
	blobs = make_blobs(n_samples=n_samples, random_state=8, center_box=(-1, 1), cluster_std=0.1)
	random_data = np.random.rand(n_samples, 2), None
	colors = 'bgrcmyk'
	data = [circles, moons, blobs, random_data]
	models = [('None', None),
			  ('Kmeans', KMeans(n_clusters=3)),
			  ('DBSCAN', DBSCAN(min_samples=3, eps=0.2)),
			  ('Agglomerative', AgglomerativeClustering(n_clusters=3, linkage='average'))]
	f = plt.figure()
	for inx, clt in enumerate(models):
		clt_name, clt_entity = clt
		for i, dataset in enumerate(data):
			x, y = dataset
			if not clt_entity:
				clt_res = [0 for item in range(len(x))]
			else:
				clt_entity.fit(x)
				clt_res = clt_entity.labels_.astype(np.int)
			f.add_subplot(len(models), len(data), inx*len(data) + i + 1)
			plt.title(clt_name)
			[plt.scatter(x[p, 0], x[p, 1], color=colors[clt_res[p]]) for p in range(len(x))]
	plt.show()


if __name__ == '__main__':
	main()
