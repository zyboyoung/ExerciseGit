import requests
import re
import json

url = 'http://kaijiang.500.com/'

def get_html(url):
	headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
	response = requests.get(url, headers = headers)
	response.encoding = 'gb2312'
	html = response.text
	return html


def html_parser(string):

	ssq_data = re.search(r'<tr id="ssq">([\d\D]*?)</tr>', string).group()
	sd_data = re.search(r'<tr id="sd">([\d\D]*?)</tr>', string).group()
	qlc_data = re.search(r'<tr id="qlc">([\d\D]*?)</tr>', string).group()

	ssq_dict = {}
	sd_dict = {}
	qlc_dict = {}

	def info_extract(data):
		periods_html = re.search(r'<td align="center">(\d*)期([\d\D]*?)</td>', data).group()
		date_html = re.search(r'<td align="center">(\d*-\d*-\d)([\d\D]*?)</td>', data).group()
		return periods_html, date_html
	
	ssq_number = re.search(r'formatResult([\d\D]*?);', ssq_data).group()
	sd_number = re.search(r'formatResult([\d\D]*?);', sd_data).group()
	qlc_number = re.search(r'formatResult([\d\D]*?);', qlc_data).group()


	ssq_dict['期号'] = delete_table(info_extract(sd_data)[0]).strip()
	ssq_dict['开奖时间'] = delete_table(info_extract(sd_data)[1]).strip()
	ssq_dict['开奖号码'] = re.findall(r'(\d{2})', ssq_number)

	sd_dict['期号'] = delete_table(info_extract(ssq_data)[0]).strip()
	sd_dict['开奖时间'] = delete_table(info_extract(ssq_data)[1]).strip()
	sd_dict['开奖号码'] = re.findall(r'(\d,\d,\d)', sd_number)[0]
	sd_dict['试机号'] = re.findall(r'(\d,\d,\d)', sd_number)[1]

	qlc_dict['期号'] = delete_table(info_extract(qlc_data)[0]).strip()
	qlc_dict['开奖时间'] = delete_table(info_extract(qlc_data)[1]).strip()
	qlc_dict['开奖号码'] = str(re.findall(r'\d{2}', qlc_number)[0:-1]) + ',' + str(re.findall(r'\d{2}', qlc_number)[-1])

	return ssq_dict, sd_dict, qlc_dict


def delete_table(string):
	delete_patt = re.compile(r'<[^>]+>',re.S)
	string = delete_patt.sub('',string)
	return string


if __name__ == '__main__':

	ssq_dict, sd_dict, qlc_dict = html_parser(get_html(url))

	result = {}
	result['双色球'] = ssq_dict
	result['福彩3D'] = sd_dict
	result['七乐彩'] = qlc_dict

	print('result = ', json.dumps(result, ensure_ascii=False, indent=4, separators=(',', ': ')))

