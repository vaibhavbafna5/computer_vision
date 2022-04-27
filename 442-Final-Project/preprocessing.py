import numpy as np
import requests
import pprint
import datetime as dt
from pymongo import MongoClient
import torch
from torch.autograd import Variable

from PIL import Image
from io import BytesIO

# url = 'https://calm-retreat-24389.herokuapp.com/process'

# resp = requests.post(url, json=data,)
# print(resp)

client = MongoClient("mongodb+srv://sunset-data-manager-admin:sunset442@cluster0-stvht.mongodb.net/test?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client['ImageMetaData']
image_collection = db['Images']
docs = list(image_collection.find({'src_id': '49772066873'}))[0]    # change this to .find({}) when working with full database

print("\nOBJECT RETURNED FROM API")
print("-------------------------\n")
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(docs)
datetime_obj = dt.datetime.utcfromtimestamp(docs['taken_at'])
date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
num_minutes_taken_at = datetime_obj.hour*60 + datetime_obj.minute
temp = docs['sunset'].split(':')
num_minutes_sunset = 12*60 + int(temp[0])*60 + int(temp[1])
print("\n################################################\n")
print("TAKEN AT:", num_minutes_taken_at, "SUNSET:", num_minutes_sunset)
print("MINUTES BEFORE SUNSET: ", str(num_minutes_sunset - num_minutes_taken_at))
print("\n################################################\n")
print("HOW LABELED DATA LOOKS:\n")
print(docs['image_url'] + ",", str(num_minutes_sunset - num_minutes_taken_at))
print("\n################################################\n")

response = requests.get(docs['image_url'], verify=False)
image = Image.open(BytesIO(response.content))
image.show()
size = 64, 64
image.thumbnail(size)
image.save("downsized_img.jpg")
arr = np.array(image)
image.show()
# print("\n########################\n")
# print("THE RGB MATRIX OF THE DOWNSIZED IMAGE\n")
# print("########################\n")
# print(arr)


# width, height = image.size
# print(width,height)
