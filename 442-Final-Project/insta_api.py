from instagram_private_api import Client, ClientCompatPatch
import datetime as dt
import requests
import time

url = 'https://calm-retreat-24389.herokuapp.com/process'

username = ''
password = ''

api = Client(username, password)
rank_token = Client.generate_uuid()
results = api.feed_tag("sunset", rank_token)
next_max_id = results.get('next_max_id')
count = 0
dummy_data = []

while next_max_id:
    for item in results['items']:
        if 'lat' in item and 'image_versions2' in item:
            datum = {
                'src_id': str(item['id']),
                'latitude': item['lat'],
                'longitude': item['lng'],
                'image_url': item['image_versions2']['candidates'][0]['url'],
                'taken_at': item['taken_at']
            }

            dummy_data.append(datum)
            taken_at = item['taken_at']

    post_data = {
        'user': 'Phillip Mathew',
        'image_data': dummy_data
    }

    if len(dummy_data) > 50:
        resp = requests.post(url, json=post_data,)
        print(resp)
        time.sleep(70)
        dummy_data = []

    results = api.feed_tag("sunset", rank_token, max_id=next_max_id)
    next_max_id = results.get('next_max_id')
