
def create_keys():
	three_char = []
	for first_item in ('A', 'T', 'C', 'G'):
		for second_item in ('A', 'T', 'C', 'G'):
			for third_item in ('A', 'T', 'C', 'G'):
				three_char.append(first_item + second_item + third_item)
	return three_char

def data_extract(filename, keys_list):
	with open(filename) as file:
		file = file.readlines()

		for i in range(len(file)):
			count_list = []
			count_dic = {}

			pre_data = file[i].split('\t')[-1]

			sum = 0.0
			for item in keys_list:
				sub_count = pre_data.count(item)
				sum += sub_count
				count_list.append(str(item) + ':' + str(sub_count))

			# print(count_list)

			for item in count_list:
				count_key = item.split(':')[0]
				count_value = item.split(':')[1]
				count_dic[count_key] = int(count_value)/sum
			print(i+1, ':')
			print(count_dic)

			# exit()

if __name__ == '__main__':
	filename = r'C:\Users\62764\Desktop\chr1_tad.seq'
	keys_list = create_keys()
	data_extract(filename, keys_list)
