import LRP2E
import random
import requests
import demjson
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

random.seed(1)

temp_li = ['平武县', '广元市', '青川县']
city_li = ['汶川县', '茂县', '北川羌族自治县', '安县', '平武县', '绵竹市', '什邡市', '都江堰市', '彭州', '青川县',
           '理县', '江油市', '广元市', '绵阳市', '德阳市']
depot_li = ['成都站', '双流县']
city_li.extend(depot_li)


def a_map_location(city, a_map_key=''):
    r = requests.get('http://restapi.amap.com/v3/geocode/geo?address={0}&output=JSON&key={1}'.format(city, a_map_key))
    json_data = r.text
    text = demjson.decode(json_data)
    try:
        location = text['geocodes'][0]['location']
        location = location.split(',', 1)
        location = [float(a) for a in location]
        return (location)
    except:
        return ('erro')


def google_map_location(city, google_map_key=''):
    url = 'https://maps.googleapis.com/maps/api/geocode/json?address={}&key={}'.format(city, google_map_key)
    r = requests.get(url)
    json_data = r.text
    text = demjson.decode(json_data)
    try:
        location = text['results'][0]['geometry']['location']
        location = [float(location['lng']), float(location['lat'])]
        return (location)
    except:
        return ('erro')


def google_map_draw(google_map_key=''):
    # center = '31.0691798,103.89736454'
    center = '31.55819106,103.92717715'
    size = '640x480'
    maptype = 'hybrid'  # 'satellite'  # 'terrain'
    markers = 'size:tiny%7C' + '%7C'.join(city_li)
    url = 'https://maps.googleapis.com/maps/api/staticmap?' \
          'center={4}&zoom=7&size={0}&scale=2&maptype={1}&format=png32' \
          '&markers={3}&key={2}' \
        .format(size, maptype, google_map_key, markers, center)
    r = requests.get(url)
    file_name = 'picture_{}.png'.format(maptype)
    with open(file_name, 'wb') as file:
        file.write(r.content)
        print(file_name, 'is downloaded.')


# google_map_draw()
# depot_loc = {depot: google_map_location(depot) for depot in depot_li}
# city_loc = {city: google_map_location(city) for city in city_li}
city_loc = {'汶川县': [103.590386, 31.476822], '茂县': [103.853522, 31.681154], '北川羌族自治县': [104.46797, 31.617202],
            '安县': [104.567187, 31.534886], '平武县': [104.555583, 32.409675], '绵竹市': [104.22075, 31.338077],
            '什邡市': [104.167501, 31.12678], '都江堰市': [103.646912, 30.988434], '彭州': [103.958014, 30.990108],
            '青川县': [105.238842, 32.575484], '理县': [103.166852, 31.436473], '江油市': [104.745823, 31.778022],
            '广元市': [105.843357, 32.435435], '绵阳市': [104.679004, 31.46746], '德阳市': [104.397894, 31.126855]}
depot_loc = {'成都站': [104.07337041, 30.69681133], '双流县': [103.923651, 30.574474]}
satellite_loc = {'A': [104.7958663, 30.98789618], 'B': [105.15353347, 31.08040717], 'C': [105.82059708, 30.9315482],
                 'D': [105.89429873, 30.00876652], 'E': [106.55353126, 31.57856261]}

DEPOT_NUM, SATELLITE_NUM, CUSTOMER_NUM = len(depot_loc), len(satellite_loc), len(city_loc)
SATELLITE_CAP = float("inf")
VEHICLE1_NUM, VEHICLE2_NUM = 10, 8
VEHICLE1_CAP, VEHICLE2_CAP = 60, 60
DEPOT = {i: [[None, None], None] for i in range(DEPOT_NUM)}
SATELLITE = {i: [[None, None], None] for i in range(max(DEPOT) + 1, max(DEPOT) + 1 + SATELLITE_NUM)}
CUSTOMER = {i: [[None, None], [random.randrange(50, 400)] * DEPOT_NUM] for i in range(max(SATELLITE) + 1, max(SATELLITE) + 1 + CUSTOMER_NUM)}
id_city_dic = {}
for i, city in zip([i for i in CUSTOMER], [i for i in city_loc]):
    CUSTOMER[i][0] = city_loc[city]
    id_city_dic[i] = city
for i, city in zip([i for i in SATELLITE], [i for i in satellite_loc]):
    SATELLITE[i][0] = satellite_loc[city]
    id_city_dic[i] = city
for i, city in zip([i for i in DEPOT], [i for i in depot_loc]):
    DEPOT[i][0] = depot_loc[city]
    id_city_dic[i] = city
print(CUSTOMER)

def loop(ratio):
    PARAMETERS = {'f': 0.5, 'pop_size': 500, 'offspring_size': 300, 'archive_size': 400, 'k': 300, 'mutt_prob': 0.3,
                  'iter_times': 100}
    INSTANCE = {'depot': DEPOT, 'satellite': SATELLITE, 'customer': CUSTOMER,
                'vehicle1_num': VEHICLE1_NUM, 'vehicle2_num': VEHICLE2_NUM,
                'vehicle1_cap': VEHICLE1_CAP, 'vehicle2_cap': VEHICLE2_CAP,
                'satellite_cap': float("inf"),}

    LRP2E.main(INSTANCE, PARAMETERS)

ratio_li = [[5, 5], [6, 4], [7, 3], [8, 2], [9, 1]]
for ratio in ratio_li:
    print(ratio)

# img = mpimg.imread('s_picture_satellite.png')
# print(img[0][1])
# plt.imshow(img, alpha=0.9)