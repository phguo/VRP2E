import LRP2E
import random
import requests
import demjson
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import os
import copy
import json
import time
import traceback
import pandas as pd
import csv
import numpy as np

random.seed(1)

temp_li = ['平武县', '广元市', '青川县']
city_li = ['汶川县', '茂县', '北川羌族自治县', '安县', '平武县', '绵竹市', '什邡市', '都江堰市', '彭州', '青川县',
           '理县', '江油市', '广元市', '绵阳市', '德阳市']
depot_li = ['成都站', '双流县']


# city_li.extend(depot_li)


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
VEHICLE1_NUM, VEHICLE2_NUM = None, None
VEHICLE1_CAP, VEHICLE2_CAP = None, None
DEPOT = {i: [[None, None], None] for i in range(DEPOT_NUM)}
SATELLITE = {i: [[None, None], float("inf")] for i in range(max(DEPOT) + 1, max(DEPOT) + 1 + SATELLITE_NUM)}
CUSTOMER = {i: [[None, None], random.randrange(100, 1000)] for i in
            range(max(SATELLITE) + 1, max(SATELLITE) + 1 + CUSTOMER_NUM)}
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

INSTANCE = {'depot': DEPOT, 'satellite': SATELLITE, 'customer': CUSTOMER,
            'vehicle1_num': VEHICLE1_NUM, 'vehicle2_num': VEHICLE2_NUM,
            'vehicle1_cap': VEHICLE1_CAP, 'vehicle2_cap': VEHICLE2_CAP,
            'satellite_cap': float("inf"), }
parameters = {'f': 0.5, 'pop_size': 500, 'offspring_size': 300, 'archive_size': 400, 'k': 300, 'mutt_prob': 0.3,
              'iter_times': 100}


def make_instance():
    ratio_li = [[5, 5], [6, 4], [7, 3], [8, 2], [9, 1]]
    for ratio in ratio_li:
        name = '_'.join([str(a) for a in ratio])
        ins = copy.deepcopy(INSTANCE)
        cus_dic = ins['customer']
        for cus in cus_dic:
            cus_dic[cus][1] = [ratio[0] * cus_dic[cus][1] / 10, ratio[1] * cus_dic[cus][1] / 10]
        depot_dic = ins['depot']
        for depot in depot_dic:
            depot_dic[depot][1] = sum(cus_dic[key][1][depot] for key in cus_dic) * 0.8
        ins['vehicle1_num'], ins['vehicle2_num'] = 4, 10
        ins['vehicle1_cap'], ins['vehicle2_cap'] = 6000, 2000
        ins['name'] = name
        json_data = json.dumps(ins, sort_keys=True, indent=2, separators=(',', ':'))
        with open('./ins_analysis_data/{}.json'.format(name), 'wt') as f:
            f.write(json_data)


def load_instance_json():
    path = './ins_analysis_data/'
    files = os.listdir(path)
    files = sorted(files)
    files = [name for name in files if not name[0] == '.']
    for file in files:
        file_path = path + file
        with open(file_path, 'r') as f:
            instance = json.load(f)
            for s in ['depot', 'satellite', 'customer']:
                instance[s] = {int(key): instance[s][key] for key in instance[s]}
            yield (instance)


def run(ins):
    PARAMETERS = {'f': 0.5, 'pop_size': 500, 'offspring_size': 300, 'archive_size': 400, 'k': 300, 'mutt_prob': 0.3,
                  'iter_times': 100}
    print('==' * 40)
    print('Solving the instance:', ins['name'])
    t1 = time.clock()
    res = LRP2E.main(ins, PARAMETERS)
    t2 = time.clock()
    res.append(t2 - t1)
    print('time consuming:', t2 - t1)
    json_data = json.dumps(res, sort_keys=True, indent=2, separators=(',', ':'))
    with open('./ins_res_stand_500iter/{}.json'.format(ins['name']), 'wt') as f:
        f.write(json_data)


def sc_send(title, content='', key=''):
    url = 'http://sc.ftqq.com/' + key + '.send?text=' + title + '&desp=' + content
    r = requests.get(url)
    if r.status_code == 200:
        return ('OK')
    else:
        return ('Opps')


def main():
    ins_name = 'BUG'
    try:
        for ins in load_instance_json():
            print(ins)
            t1 = time.clock()
            ins_name = ins['name']
            run(ins)
            t2 = time.clock()
            sc_send('{}运行完毕'.format(ins_name), str((t2 - t1)) + '秒')
    except:
        title = ins_name
        content = traceback.format_exc()
        sc_send(title, content)


def read_res():
    path = './ins_res_stand/'
    files = os.listdir(path)
    files = sorted(files, key=lambda file_name: file_name)
    files = [name for name in files if not name[0] == '.']

    for file in files:
        file_path = path + file
        with open(file_path, 'r') as f:
            res = json.load(f)
            yield (file, res)


def write_res_analysis_csv():
    headers = ['instance_name', 'non_dominated_solution_num', 'time_consuming',
               'min_obj1', 'mean_obj1', 'median_obj1', 'max_obj1',
               'min_obj2', 'mean_obj2', 'median_obj2', 'max_obj2',
               'min_obj3', 'mean_obj3', 'median_obj3', 'max_obj3',
               'evaluation1', 'evaluation2', 'evaluation3']
    rows = []
    for ins_name, res_li in read_res():
        obj_values_frame = []
        for res in res_li[:-1]:
            obj_values = res[3]
            obj_values_frame.append(obj_values)
        data_frame = pd.DataFrame(obj_values_frame)
        d = data_frame.describe()
        row = (ins_name[:-5], len(res_li), res_li[-1],
               d[0]['min'], d[0]['mean'], d[0]['50%'], d[0]['max'],
               d[1]['min'], d[1]['mean'], d[1]['50%'], d[1]['max'],
               d[2]['min'], d[2]['mean'], d[2]['50%'], d[2]['max'])
        rows.append(row)

    with open('ins_res_analysis.csv', 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


def make_single_depot_instance():
    for ins in load_instance_json():
        new_ins1 = copy.deepcopy(ins)
        new_ins2 = copy.deepcopy(ins)
    return (new_ins1, new_ins2)


def draw_boxplot():
    for obj_indx in range(3):
        obj = []
        for file_name, res_li in read_res():
            li = []
            if obj_indx != 1:
                for res in res_li[:-1]:
                    li.append(abs(res[3][obj_indx]))
            else:
                for res in res_li[:-1]:
                    li.append(abs(res[3][obj_indx]) / (len(depot_loc) * len(city_loc)))
            obj.append(li)

        meanlineprops = dict(linestyle='-', linewidth=1, color='red')
        medianprops = dict(linestyle='-', linewidth=2, color='black')
        whiskerprops = dict(linestyle='--')
        labels = ['5:5', '6:4', '7:3', '8:2', '9:1']
        fig2, ax2 = plt.subplots()
        ax2.set_title('F{}'.format(str(obj_indx + 1)))
        ax2.boxplot(obj, notch=1, medianprops=medianprops, widths=0.3, showfliers=0, patch_artist=1,
                    whiskerprops=whiskerprops, meanprops=meanlineprops, showmeans=True, meanline=True, labels=labels)

        plt.savefig('{}.pdf'.format(str(obj_indx + 1)), bbox_inches='tight',  transparent=True, pad_inches=0.1)
        plt.show()


if __name__ == '__main__':
    # make_instance()
    # main()
    # write_res_analysis_csv()
    draw_boxplot()

# img = mpimg.imread('s_picture_satellite.png')
# print(img[0][1])
# plt.imshow(img, alpha=0.9)
