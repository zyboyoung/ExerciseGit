# 更新所有可以更新的Python库
import os
from pip._internal.utils.misc import get_installed_distributions
import time
from functools import wraps


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
			print(func.__name__, str(cost_time // 60), 'min', '{:.2f}'.format(cost_time % 60), 's')
		print('-' * 8)
		return result

	return wrapper


@caltime
def main():
	out_list = get_installed_distributions()

	for out_dist in out_list:
		project = out_dist.project_name
		print(project + '\n')
		action = os.popen('python -m pip install --upgrade ' + project)
		result = action.read()
		print(result)

	print('End')


if __name__ == '__main__':
	main()
