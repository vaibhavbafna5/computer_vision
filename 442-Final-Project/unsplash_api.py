# unsplash_api.py
# Gets sunset images from Unsplash api
# (https://unsplash.com/documentation#get-a-collection)
# Then pushes image data to database

# pip install python-unsplash

# import urllib.request
# from PIL import Image
from datetime import datetime
import time
import numpy as np
from urllib import *
# import cv2
import requests
import json

url = 'https://api.unsplash.com/search/photos'
geocoding_api_key = 'AIzaSyC6wN-o_zGbRaq2AC_xWvK_K8KtTL6Apj0'
db_url = 'https://calm-retreat-24389.herokuapp.com/process'

izzy_client_id = "MzueET6jIaczg2WfcJMtb69IiKL98HEyRUa9zogIe0c"
# client_secret = "0i5hsi_776Zq8GXFjXWp9Zy3ZE6-1m9p5hEQTvxLYx8"
rohit_client_id = "t6eWnK2pzQjKEOOj2a1C1MFw1pcay6S3i4Q6lYwWpKU"
ashley_client_id = "JOe8y8cH88EVznybOYA5trAf_F9_wedXyXHGHeRCw3k"
v_client_id = "X5btm0GoUcq3Jb1lRx67UF61KN3C9ExMaT-JGx8O1rU"
phillip_client_id = "gPE33rgSg_cqCarOYHqrcrPerzQ2UB0LAfVOTguieA0"

# returns latitude, longitude if valid, else returns 0, 0
# no longer using this function due to difficulty getting accurate location from Unsplash API
def get_lat_lng(city):
	url = 'https://maps.googleapis.com/maps/api/geocode/json?'
	geocoding_data = requests.get(
        url, params={'key': geocoding_api_key, 'address': city}).json()
	if geocoding_data['status'] == 'OK':
		results = geocoding_data['results'][0]
		lat = results['geometry']['location']['lat']
		lng = results['geometry']['location']['lng']
		return lat, lng
	return 0, 0


def get_unsplash_data(client_id, page_number):
	print("Page #: ", page_number)
	unsplash_data = requests.get(
		url, params={'client_id': client_id, 'query': 'sunset', 'page': page_number, 'per_page':49 })
	unsplash_data = unsplash_data.json()
	
	photos_results = unsplash_data.get('results')
	
	sunset_picture_list = []
	for photo in photos_results:
		sunset_dict = {}
		sunset_dict['src_id'] = photo.get('id')
		print("photo id = ", photo.get('id'))
		# print("\n photo info \n", photo)
		sunset_dict['image_url'] = photo.get('full')
		if (sunset_dict['src_id'] != None):
			# "2017-08-26T23:13:06-04:00"
			pic_url = 'https://api.unsplash.com/photos'
			add_to_url = '/' + photo.get('id')
			pic_data = requests.get(pic_url + add_to_url, params={'client_id': client_id}).json()

			lat = pic_data['location']['position']['latitude']
			lng = pic_data['location']['position']['longitude']
			# print("PIC DATA = \n", pic_data)
			# print("pic data id = ", pic_data['id'])
			# print(lat, lng)
			
			if lat is not None and lng is not None:
				# ** Should we be overwriting the src_id here that we already set above (Line 46)?
				sunset_dict['image_url'] = pic_data.get('urls').get('full')
				sunset_dict['latitude'] = float(lat)
				sunset_dict['longitude'] = float(lng)
				date_time = photo.get('created_at')
				date_taken = datetime.strptime(date_time, "%Y-%m-%dT%H:%M:%S%z")
				date_taken = time.mktime(date_taken.timetuple())
				sunset_dict['taken_at'] = int(date_taken)
				sunset_picture_list.append(sunset_dict)
				print("LOCATION SUCCESS")
			else:
				print("no location")

	data = {}
	data['user'] = "rohit&izzy"
	data['image_data'] = sunset_picture_list
	# print("\nData is = ", data) 
	
	if len(sunset_picture_list) != 0:
		print("Number of pics sent to DB: ", len(sunset_picture_list))
		resp = requests.post(db_url, json=data,)
		print("Response from DB: ", resp)
		time.sleep(120)
	else:
		print("## Data is empty")
	return None

def main():
	izzy_client_id = "MzueET6jIaczg2WfcJMtb69IiKL98HEyRUa9zogIe0c"
	# client_secret = "0i5hsi_776Zq8GXFjXWp9Zy3ZE6-1m9p5hEQTvxLYx8"
	rohit_client_id = "t6eWnK2pzQjKEOOj2a1C1MFw1pcay6S3i4Q6lYwWpKU"
	ashley_client_id = "JOe8y8cH88EVznybOYA5trAf_F9_wedXyXHGHeRCw3k"
	v_client_id = "X5btm0GoUcq3Jb1lRx67UF61KN3C9ExMaT-JGx8O1rU"
	phillip_client_id = "gPE33rgSg_cqCarOYHqrcrPerzQ2UB0LAfVOTguieA0"
	# for num in range(10):
	num = 11
	get_unsplash_data(izzy_client_id, num)
	num = num + 1
	get_unsplash_data(rohit_client_id, num)
	num = num + 1
	get_unsplash_data(ashley_client_id, num)
	num = num + 1
	get_unsplash_data(v_client_id, num)
	num = num + 1
	get_unsplash_data(phillip_client_id, num)

	return None

if __name__ == "__main__":
    main()