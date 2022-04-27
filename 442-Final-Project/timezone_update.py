import numpy as np
import requests
import pprint
import datetime as dt
from pymongo import MongoClient
from timezonefinder import TimezoneFinder
from pytz import timezone
import pytz
import iso8601

client = MongoClient("mongodb+srv://sunset-data-manager-admin:sunset442@cluster0-stvht.mongodb.net/test?retryWrites=true&w=majority&ssl=true&ssl_cert_reqs=CERT_NONE")
db = client['ImageMetaData']
image_collection = db['Images_2']

docs = list(image_collection.find())

tf = TimezoneFinder()
utc = pytz.utc
for doc in docs:
    if doc['taken_at'] and doc['sunset']:
        tz = tf.timezone_at(lng=doc['longitude'], lat=doc['latitude']) # returns something like 'Europe/Berlin'
        try:
            tz = timezone(tz)
            utc_dt = utc.localize(dt.datetime.utcfromtimestamp(doc['taken_at']))    # this is taken_at in utc

            actual_taken_at = utc_dt.astimezone(tz)     # this is taken_at in the actual timezone

            URL = "https://api.sunrise-sunset.org/json?lat=" + str(doc['latitude']) +"&lng=" + str(doc['longitude']) + "&date=" + str(actual_taken_at.strftime("%Y-%m-%d") + "&formatted=0")

            response = requests.get(url = URL)
            new_sunset = response.json()['results']['sunset']   # this is the sunset in UTC based off the actual taken_at

            temp = new_sunset.split(':')
            sunset_unix_time = iso8601.parse_date(new_sunset).timestamp()

            difference = int(sunset_unix_time - doc['taken_at']) // 60
            print("DIFFERENCE: ", difference)

            oc = image_collection.find_one_and_update(
                {'src_id': doc['src_id']},
                {"$set":
                    {"minutes_until_sunset": difference}
                },upsert=False
            )
        except:
            pass
