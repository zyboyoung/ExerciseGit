# 每个房子有一定数量的现金。当有两个相邻的房子在同一晚被劫时，安保系统才会自动触发。 现在给你一个正整数数组表示每家现金数，请求出这一晚你能在不触发安保系统时抢到的最大金额。
def solution(money: list):
	n = len(money)
	if n == 0:
		return 0
	elif n == 1:
		return money[0]
	else:
		values = [float('-inf')] * n
		values[0] = money[0]
		values[1] = max(money[0], money[1])

		for i in range(2, n):
			values[i] = max(values[i - 2] + money[i], values[i - 1])
		return values[n - 1]
