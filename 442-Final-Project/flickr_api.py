# First, you should install flickrapi
# pip install flickrapi

import flickrapi
import urllib.request
from PIL import Image
from scipy import misc
from datetime import datetime
import time
import numpy as np
from urllib import *
import cv2
import requests
import json

# Flickr api access key 

db_url = 'https://calm-retreat-24389.herokuapp.com/process'

sunset_picture_list = []    
keyword = 'sunset'
PARAMS = ["url_c", "datetaken", "geo"]
extras = ','.join(PARAMS)
page_number = 429
# flickr_key = "454ee4be62cafca4c2958a5e6a67b58f" # todo fix
# Created new flickr_key below
flickr_key = "2c3b1b3d0507bcbcf8f4bbf329be5040"
auth = '72157713833405946-2f1cac925b02f075api_sig=bc77d3366ccf429dac460afcab4c0e45'
base_url = 'https://api.flickr.com/services/rest/'

while page_number < 700:
    time.sleep(60)
    sunset_picture_list = [] 
    params = {}
    params['method'] = 'flickr.photos.search'
    params['api_key'] = flickr_key
    # params['auth_token'] = auth
    params['format'] = 'json'
    params['tags'] = ['sunset']
    params['tag_mode'] = 'all'
    params['per_page'] = 500
    params['page'] = page_number
    extras = ["url_c", "date_taken", "geo"]
    # Added the parameter below ('nojsoncallback')
    params['nojsoncallback'] = 1
    extras = ','.join(extras)
    params['extras'] = extras
    try:
        # flickr_url = "https://www.flickr.com/services/rest/?method=flickr.photos.search&api_key=454ee4be62cafca4c2958a5e6a67b58f&tags=sunset&accuracy=10&content_type=1&geo_context=2&extras=date_taken&per_page=500&page=" + page_number + "&format=rest&auth_token=72157713833405946-2f1cac925b02f075&api_sig=4834459686ea8b944f7a94ff114ef72b"
        flickr_response = requests.get(base_url, params=params)
        # print("response 1 ", flickr_response)
        # print(flickr_response.text)
        response_text = flickr_response.text
        # print(response_text[14:-1])
        # print("@@@@@@\n \n")
        parsed_response = json.loads(response_text)
        # print("\n \n \n$$$$$$")

        # print("response 2", parsed_response)
        photos = parsed_response['photos']['photo']
        # filename = "out" + str(page_number) + ".txt"
        # with open(filename, 'w') as f:
        #     f.write(str(photos))
        # print("\n\n\n")
        # print("flickr repsonse photos", photos)
    except:
        print("EXCEPTION: broke here 1")
        break

    print("Page #: ", page_number)
    urls = []
    for i, photo in enumerate(photos):
        sunset_dict = {}
        # if i % 10 == 0:
            # print (i)
        # urls.append(url)

        # print("TYPE: ", photo.get('latitude').type())
        
        # get 50 urls
        if i > 1000:
            break
        if photo.get('latitude') != 0 and photo.get('longitude') != 0 and photo.get('datetaken') != None and photo.get('url_c') != None and photo.get('datetakenunknown') == '0':            url = photo['url_c']
            try:
                # urllib.request.urlretrieve(url, '00001.jpg')
                resp = urllib.request.urlopen(url)
                # print(resp)
                image = np.asarray(bytearray(resp.read()), dtype="uint8")
                # print(image)
                image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            except:
                print("EXCEPTION: broke here 2")
                break

            # print(image.shape)

            # Resize the image and overwrite it
            # image = Image.open('00001.jpg') 
            image = np.resize(image, (256, 256, 3))
            sunset_dict['src_id'] = str(photo.get('id'))
            # print("id is = ", sunset_dict['src_id'])
            sunset_dict['image_url'] = url
            sunset_dict['latitude'] = float(photo.get('latitude'))
            sunset_dict['longitude'] = float(photo.get('longitude'))
            date_taken = photo.get('datetaken')
            date_taken = datetime.strptime(date_taken, "%Y-%m-%d %H:%M:%S")
            date_taken = time.mktime(date_taken.timetuple())
            # print("unix time is = ", date_taken)
            sunset_dict['taken_at'] = int(date_taken)
            lat = photo.get('latitude')
            long_loc = photo.get('longitude')
            date = photo.get('datetaken')
            date = date[0:10]

            # URL = "https://api.sunrise-sunset.org/json?lat=" + str(lat) +"&lng=" + str(long_loc) + "&date=" + date

            # response = requests.get(url = URL)
            # print("response is ", response.json())
            # sunset_dict['sunset'] = response.json()['results']['sunset']
            # sunset_dict['sunrise'] = response.json()['results']['sunrise']
           
            sunset_picture_list.append(sunset_dict)
            # print("appended pic")

    page_number += 1

    print("Length of images:")
    print(len(sunset_picture_list))
# with open("final_output.txt", 'w') as f:
#     f.write(str(sunset_picture_list))

# for sunset in sunset_picture_list:
#     if i % 10 == 0:
#         print (sunset)
    data = {}
    data['user'] = "izzy&ashley"
    data['image_data'] = sunset_picture_list
# print(data) 
    resp = requests.post(db_url, json=data,)
    print(resp)