from functools import partial


class Card:
	def __init__(self, rank, suit):
		self.suit = suit
		self.rank = rank
		self.hard, self.soft = self._points()

	def _points(self):
		return 0, 0


class NumberCard(Card):
	def _points(self):
		return int(self.rank), int(self.rank)


class AceCard(Card):
	def _points(self):
		return 1, 11


class FaceCard(Card):
	def _points(self):
		return 10, 10


class Suit:
	def __init__(self, name, symbol):
		self.name = name
		self.symbol = symbol


Club, Diamond, Heart, Spade = Suit('Club', '♧'), Suit('Diamond', '♢'), Suit('Heart', '♡'), Suit('Spade', '♤')


# 构建牌组的方法1：使用elif和映射
def card1(rank, suit):
	if rank == 1:
		return AceCard('A', suit)
	elif 1 < rank < 11:
		return NumberCard(str(rank), suit)
	elif 11 <= rank < 14:
		name = {11: 'J', 12: 'Q', 13: 'K'}[rank]
		return FaceCard(name, suit)
	else:
		raise Exception('Rank out of range')


# 构建牌组的方法2：只使用映射，用到了dict.get(key, default)以及偏函数partial
def card2(rank, suit):
	part_class = {
		1: partial(AceCard, 'A'),
		11: partial(FaceCard, 'J'),
		12: partial(FaceCard, 'Q'),
		13: partial(FaceCard, 'K'),
	}.get(rank, partial(NumberCard, str(rank)))
	return part_class(suit)


# 工厂模式的流畅API设计
class CardFactory:
	def rank(self, rank):
		self.class_, self.rank_str = {
			1: (AceCard, 'A'),
			11: (AceCard, 'J'),
			12: (AceCard, 'Q'),
			13: (AceCard, 'K'),
		}.get(rank, (NumberCard, str(rank)))
		return self

	def suit(self, suit):
		return self.class_(self.rank_str, suit)


card3 = CardFactory()
deck3 = [card3.rank(r).suit(s) for r in range(1, 14) for s in [Club, Diamond, Heart, Spade]]

deck = [card2(rank, suit) for rank in range(1, 14) for suit in [Club, Diamond, Heart, Spade]]

for card in deck3:
	print(card.rank, card.suit.symbol)
