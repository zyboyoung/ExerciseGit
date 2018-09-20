from functools import wraps
import time
import json


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
		print(func.__name__, (end - start) * 1000, 'ms')
		print('-' * 8)
		return result

	return wrapper


@caltime
def reverse_words(s):
	"""
	反转字符串中的单词
	:param s: str
	:return: str
	"""
	return ' '.join(i[::-1] for i in s.split())


@caltime
def number2words(num):
	"""
	Convert a non-negative integer to its english words representation
	:param num: int
	:return: str
	"""
	to19 = 'One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve \
	Thirteen Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen'.split()
	tens = 'Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety'.split()

	def words(n):
		if n < 20:
			return to19[n - 1:n]
		if n < 100:
			return [tens[int(n / 10) - 2]] + words(n % 10)
		if n < 1000:
			return [to19[int(n / 100) - 1]] + ['Hundred'] + words(n % 100)
		for p, w in enumerate(('Thousand', 'Million', 'Billion'), 1):
			if n < 1000 ** (p + 1):
				return words(int(n / 1000) ** p) + [w] + words(n % 1000 ** p)

	return ' '.join(words(num)) or 'Zero'


@caltime
def two_sum(nums: list, target: int) -> list:
	# return indices of the two numbers such that they add up to a specific target
	if len(nums) <= 1:
		return []
	buff_dict = {}
	for i in range(len(nums)):
		if nums[i] in buff_dict:
			return [buff_dict[nums[i]], i]
		else:
			buff_dict[target - nums[i]] = i


class ListNode:
	def __init__(self, x):
		self.val = x
		self.next = None


@caltime
def add_two_numbers(l1: ListNode, l2: ListNode) -> ListNode:
	carry = 0
	root = n = ListNode(0)
	while l1 or l2 or carry:
		v1 = v2 = 0
		if l1:
			v1 = l1.val
			l1 = l1.next
		if l2:
			v2 = l2.val
			l2 = l2.next
		carry, val = divmod(v1+v2+carry, 10)
		n.next = ListNode(val)
		n = n.next
	return root.next


@caltime
def length_longest_substring(s: str) -> int:
	start = max_length = 0
	used_char = {}

	for i in range(len(s)):
		if s[i] in used_char and start <= used_char[s[i]]:
			start = used_char[s[i]] + 1
		else:
			max_length = max(max_length, i - start + 1)

		used_char[s[i]] = i

	return max_length


@caltime
def find_median_sorted_arrays(nums1: list, nums2: list) -> float:
	total = []
	s = len(nums1) + len(nums2)
	m, n = 0, 0

	for i in range(s):
		if m == len(nums1):
			total.append(nums2[n])
			n += 1
		elif n == len(nums2):
			total.append(nums1[m])
			m += 1
		else:
			if nums1[m] <= nums2[n]:
				total.append(nums1[m])
				m += 1
			else:
				total.append(nums2[n])
				n += 1

	return float(total[s//2]) if s % 2 == 1 else (total[s//2] + total[s//2 - 1]) / 2


def share_things(things: int, basket: int) -> int:
	# M个相同物体放到N个相同篮子里有多少种放法，允许有篮子不放
	if things == 0 or basket == 1:
		# 迭代停止条件
		return 1
	elif basket > things:
		# 当篮子少于物体时，总有空篮子，去掉最少数目的空篮子
		return share_things(things, things)
	else:
		# 分为两种情况，1. 至少有一个空篮子；2. 没有空篮子，每个篮子中至少有一个物体
		return share_things(things, basket - 1) + share_things(things - basket, basket)


@caltime
def sort_height(line: str) -> json:
	# 用一个数组表示一群正在排队的小学生，每个小学生用一对整数来表示 [height, k], height 表示这个小学生的身高，k 表示这个小学生前面应该有 k 个人的身高 >= 他
	# 读取数据，转化为json格式[[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
	person_list = json.loads(line)
	# 优先按照height，次优先按照k重排序
	person_list = sorted(person_list, key=lambda item: (-item[0], item[1]))
	# 排序后，新建空列表，按照k值插入，对于k值相同的，或者k值和height都小的，由于已经排序，能够保证height大的一直会先insert，因此会在后面
	res = []
	for person in person_list:
		res.insert(person[1], person)

	return json.dumps(res, separators=(',', ':'))


@caltime
def find_median_sorted_arrays(nums1, nums2):
	# 给定两个大小为m和n的有序数组nums1和nums2，找出这两个有序数组的中位数，时间复杂度为O(log(m+n))。
	def kth(a, b, k):
		if not a:
			return b[k]
		if not b:
			return a[k]

		mid_index_a = len(a) // 2
		mid_index_b = len(b) // 2
		mid_a = a[mid_index_a]
		mid_b = b[mid_index_b]

		# 当k大于a和b的中位数索引之和
		if mid_index_a + mid_index_b < k:
			if mid_a > mid_b:
				return kth(a, b[mid_index_b + 1:], k - mid_index_b - 1)
			else:
				return kth(a[mid_index_a + 1:], b, k - mid_index_a - 1)
		else:
			if mid_a > mid_b:
				return kth(a[:mid_index_a], b, k)
			else:
				return kth(a, b[:mid_index_b], k)

	n = len(nums1) + len(nums2)
	if n % 2 == 1:
		return kth(nums1, nums2, n // 2)
	else:
		return (kth(nums1, nums2, n // 2) + kth(nums1, nums2, n // 2 - 1)) / 2


if __name__ == '__main__':
	print(find_median_sorted_arrays([1, 2],[3, 4]))
