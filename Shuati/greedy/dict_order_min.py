# 给定长度为N的字符串S，每次从S的头部或者尾部删除一个字符，加到空字符串T的尾部，构造一个长度为N的字符串T，目标是构造字典序尽可能小的字符串T
def dict_order_min(s: str) -> str:
	a, b = 0, len(s) - 1
	t = ''
	while a <= b:
		left = False
		for i in range(b - a + 1):
			if s[a + i] < s[b - i]:
				left = True
				break
			elif s[a + i] > s[b - i]:
				left = False
				break
		if left:
			t += s[a]
			a += 1
		else:
			t += s[b]
			b -= 1
	return t


# 不断取s开头和末尾中较小得一个字符，二者相等情况下，比较下一个字符
if __name__ == '__main__':
	print(dict_order_min('ACDBCB'))
