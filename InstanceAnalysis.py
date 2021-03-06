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
import key
import networkx as nx

random.seed(1)

temp_li = ['平武县', '广元市', '青川县']
city_li = ['汶川县', '茂县', '北川羌族自治县', '安县', '平武县', '绵竹市', '什邡市', '都江堰市', '彭州', '青川县',
           '理县', '江油市', '广元市', '绵阳市', '德阳市']
depot_li = ['成都站', '双流县']
satellite_loc = {'A': [104.7958663, 30.98789618], 'B': [105.15353347, 31.08040717], 'C': [105.82059708, 30.9315482],
                 'D': [105.89429873, 30.00876652], 'E': [106.55353126, 31.57856261]}
satellite_loc['E'] = [103.55353126, 30.57856261]
li = [str(satellite_loc[key][1]) + ',' + str(satellite_loc[key][0]) for key in satellite_loc]
# city_li.extend(li)
depot_loc = {'成都站': [104.07337041, 30.69681133], '双流县': [103.923651, 30.574474]}
li2 = [str(depot_loc[key][1]) + ',' + str(depot_loc[key][0]) for key in depot_loc]
print(city_li)


def a_map_location(city, a_map_key=key.a_map_key):
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


def google_map_location(city, google_map_key=key.google_map_key):
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


def google_map_draw(google_map_key=key.google_map_key):
    # center = '31.0691798,103.89736454'
    # center = '31.55819106,103.92717715'
    size = '640x480'
    maptype = 'terrain'  # 'hybrid'  # 'satellite'  # 'terrain'
    markers = 'size:small%7Ccolor:red%7Clabel:.%7C' + '%7C'.join(city_li)
    markers2 = 'size:small%7Ccolor:blue%7Clabel:.%7C' + '%7C'.join(li)
    markers3 = 'size:mid%7Ccolor:gray%7Clabel:.%7C' + '%7C'.join(li2)
    print(markers)
    url = 'https://maps.googleapis.com/maps/api/staticmap?' \
          'size={0}' \
          '&scale=2' \
          '&maptype={1}' \
          '&format=png32' \
          '&markers={3}&markers={4}&markers={5}' \
          '&key={2}' \
        .format(size, maptype, google_map_key, markers, markers2, markers3)
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
satellite_loc['E'] = [103.55353126, 30.57856261]

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
    # print('{},{}'.format(i, city))
    CUSTOMER[i][0] = city_loc[city]
    id_city_dic[i] = city
for i, city in zip([i for i in SATELLITE], [i for i in satellite_loc]):
    # print('{},{}'.format(i, city))
    SATELLITE[i][0] = satellite_loc[city]
    id_city_dic[i] = city
for i, city in zip([i for i in DEPOT], [i for i in depot_loc]):
    # print('{},{}'.format(i, city))
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
        with open('./ins_analysis_data_/{}.json'.format(name), 'wt') as f:
            f.write(json_data)


def load_instance_json():
    path = './ins_analysis_data_/'
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
    res = LRP2E.main(ins, PARAMETERS, 0)
    t2 = time.clock()
    res.append(t2 - t1)
    print('time consuming:', t2 - t1)
    json_data = json.dumps(res, sort_keys=True, indent=2, separators=(',', ':'))
    with open('./ins_res_stand_/{}.json'.format(ins['name']), 'wt') as f:
        f.write(json_data)


def sc_send(title, content='', key=key.sc_key):
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
        # sc_send(title, content)


def read_res(separate):
    if separate:
        path = './ins_res_stand_separate_/'
    else:
        path = './ins_res_stand_/'
    files = os.listdir(path)
    files = sorted(files, key=lambda file_name: file_name)
    files = [name for name in files if not name[0] == '.']
    for file in files:
        file_path = path + file
        with open(file_path, 'r') as f:
            res = json.load(f)
            yield (file, res)


# def write_res_analysis_csv():
#     headers = ['instance_name', 'non_dominated_solution_num', 'time_consuming',
#                'min_obj1', 'mean_obj1', 'median_obj1', 'max_obj1',
#                'min_obj2', 'mean_obj2', 'median_obj2', 'max_obj2',
#                'min_obj3', 'mean_obj3', 'median_obj3', 'max_obj3',
#                'evaluation1', 'evaluation2', 'evaluation3']
#     rows = []
#     for ins_name, res_li in read_res():
#         obj_values_frame = []
#         for res in res_li[:-1]:
#             obj_values = res[3]
#             obj_values_frame.append(obj_values)
#         data_frame = pd.DataFrame(obj_values_frame)
#         d = data_frame.describe()
#         row = (ins_name[:-5], len(res_li), res_li[-1],
#                d[0]['min'], d[0]['mean'], d[0]['50%'], d[0]['max'],
#                d[1]['min'], d[1]['mean'], d[1]['50%'], d[1]['max'],
#                d[2]['min'], d[2]['mean'], d[2]['50%'], d[2]['max'])
#         rows.append(row)
#
#     with open('ins_res_analysis.csv', 'w') as f:
#         f_csv = csv.writer(f)
#         f_csv.writerow(headers)
#         f_csv.writerows(rows)


def draw_boxplot(obj, title, file_name, ylabel, notch, showfliers, showmeans, meanline):  # title file_name y_label
    meanlineprops = dict(linestyle='-', linewidth=1, color='red')
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    whiskerprops = dict(linestyle='--')
    flierprops = dict(marker='x', markerfacecolor='k', markersize=5, linestyle='none')
    meanprops = dict(marker='d', markeredgecolor='r', markerfacecolor='red', markersize=3)

    labels = ['5:5', '5:5', '6:4', '6:4', '7:3', '7:3', '8:2', '8:2', '9:1', '9:1']
    fig2, ax = plt.subplots(nrows=1, ncols=1)
    ax.set_title(title)
    # plt.plot([1,  3,  5,  7,  9], [np.mean(obj[i]) for i in range(len(obj)) if i % 2 == 0], color='k')
    # plt.plot([2,  4,  6,  8,  10], [np.mean(obj[i]) for i in range(len(obj)) if i % 2 != 0], color='k')

    bplot = ax.boxplot(obj, notch=notch, medianprops=medianprops, widths=0.45, showfliers=showfliers, flierprops=flierprops,
                       patch_artist=1,
                       whiskerprops=whiskerprops, meanprops=meanlineprops, showmeans=showmeans, meanline=meanline)

    colors = ['lightgray', 'white'] * 5
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_xlabel('p1 : p2')
    ax.set_ylabel(ylabel)
    plt.setp(ax, xticks=[i + 1 for i in range(len(obj))], xticklabels=labels)
    plt.savefig(file_name, bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()


def obj_boxplot(notch=1, showfliers=1, showmeans=1, meanline=1):
    for obj_indx in range(3):
        obj = []
        for i in [0, 1]:
            for file_name, res_li in read_res(i):
                li = []
                if obj_indx != 1:
                    for res in res_li[:-1]:
                        li.append(abs(res[3][obj_indx]))
                else:
                    for res in res_li[:-1]:
                        li.append(abs(res[3][obj_indx]) / (len(depot_loc) * len(city_loc)))
                obj.append(li)
        temp_li = []
        for i in range(0, 5):
            temp_li.append(obj[i])
            temp_li.append(obj[i + 5])
        obj = temp_li[:]
        title = 'F{}'.format(str(obj_indx + 1))
        file_name = '{}.pdf'.format(str(obj_indx + 1))
        ylabel = 'F{} value'.format(str(obj_indx + 1))
        draw_boxplot(obj, title, file_name, ylabel,
                     notch=notch, showfliers=showfliers, showmeans=showmeans, meanline=meanline)


def num_boxplot(notch=0, showfliers=0, showmeans=0, meanline=0):
    a = [('S', 'S.pdf', '|S|'), ('U', 'U.pdf', '|U|'), ('V', 'V.pdf', '|V|')]
    for j, triple in enumerate(a):
        if j == 0:
            obj = []
            for i in [0, 1]:
                for file_name, res_li in read_res(i):
                    li = []
                    for res in res_li[:-1]:
                        u_num = len([key for key in res[2]])
                        li.append(u_num)
                    obj.append(li)
        elif j == 1 or j == 2:
            obj = []
            for i in [0, 1]:
                for file_name, res_li in read_res(i):
                    li = []
                    for res in res_li[:-1]:
                        u_num = 0
                        for key in res[j]:
                            u_num += len(res[j][key])
                        li.append(u_num)
                    obj.append(li)
        temp_li = []
        for i in range(0, 5):
            temp_li.append(obj[i])
            temp_li.append(obj[i + 5])

        obj = temp_li[:]
        title, file_name, ylabel = triple
        draw_boxplot(obj, title, file_name, ylabel,
                     notch=notch, showfliers=showfliers, showmeans=showmeans, meanline=meanline)

        # obj = []
        # for i in [0, 1]:
        #     for file_name, res_li in read_res(i):
        #         li = []
        #         for res in res_li[:-1]:
        #             u_num = 0
        #             # u_num = len([key for key in res[2]])
        #             for key in res[2]:
        #                 u_num += len(res[2][key])
        #             li.append(u_num)
        #         obj.append(li)
        # temp_li = []
        # for i in range(0, 5):
        #     temp_li.append(obj[i])
        #     temp_li.append(obj[i + 5])
        # obj = temp_li[:]
        # title = 'satellite'
        # file_name = 'satellite.pdf'
        # ylabel = 'used satellite'
        # draw_boxplot(obj, title, file_name, ylabel,
        #              notch=notch, showfliers=showfliers, showmeans=showmeans, meanline=meanline)


def scatter_ins():
    for ins in load_instance_json():
        # print(ins['satellite'])
        customer_x = [ins['customer'][key][0][0] for key in ins['customer']]
        customer_y = [ins['customer'][key][0][1] for key in ins['customer']]
        satellite_x = [ins['satellite'][key][0][0] for key in ins['satellite']]
        satellite_y = [ins['satellite'][key][0][1] for key in ins['satellite']]
        depot_x = [ins['depot'][key][0][0] for key in ins['depot']]
        depot_y = [ins['depot'][key][0][1] for key in ins['depot']]
        # [106.55353126, 31.57856261]
        # satellite_x[-2], satellite_y[-2] = 103.55353126, 30.57856261
        plt.scatter(customer_x, customer_y,  marker='x', s=50, c='k', lw=1,label='demand nodes')
        plt.scatter(satellite_x, satellite_y,  marker='o', s=50, c='w', edgecolors='k',label='satellite nodes')
        plt.scatter(depot_x, depot_y,  marker='D', s=50, c='k', edgecolors='k',label='plants')
        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.legend()
        plt.savefig('fffff.pdf', bbox_inches='tight', transparent=True, pad_inches=0.1)
        plt.show()
        # li = [(key, ins['customer'][key]) for key in ins['customer']]
        # for a in li:
        #     print('{},{},{},{}'.format(a[0], a[1][0][0], a[1][0][1], sum(a[1][1])))
        #
        # break


def draw_route():
    pos = {}
    for ins in load_instance_json():
        pos.update(ins['depot'])
        pos.update(ins['satellite'])
        pos.update(ins['customer'])
        for key in pos:
            pos[key] = pos[key][0]
        print(pos)
        break

    separate, start = 0, 22
    for ins_name, res_li in read_res(separate):
        for res in res_li[start:]:
            depot_satellite_route = res[1]
            satellite_customer_route = res[2]
            print(depot_satellite_route)
            print(satellite_customer_route)
            edges1 = []
            for key in depot_satellite_route:
                route_li = depot_satellite_route[key]
                for li in route_li:
                    for i in range(len(li)):
                        if i == 0:
                            edges1.append([int(key), li[i]])
                            edges1.append([li[i], li[i+1]])
                        elif i == len(li) - 1:
                            edges1.append([li[i], int(key)])
                        else:
                            edges1.append([li[i], li[i+1]])
            edges2 = []
            for key in satellite_customer_route:
                route_li = satellite_customer_route[key]
                for li in route_li:
                    for i in range(len(li)):
                        if i == 0:
                            edges2.append([int(key), li[i]])
                            edges2.append([li[i], li[i + 1]])
                        elif i == len(li) - 1:
                            edges2.append([li[i], int(key)])
                        else:
                            edges2.append([li[i], li[i + 1]])
            break
        break

    print(edges1)
    print(edges2)
    G = nx.Graph()
    G.add_nodes_from([key for key in pos])
    G.add_edges_from(edges1, alpha=0.5)
    G.add_edges_from(edges2, alpha=0.5)
    colors = ['silver'] * 2 + ['skyblue'] * 5 + ['w'] * 15
    nodesize = [150] * 2 + [200] * 5 + [100] * 15
    nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=nodesize, edge_color='r', widths=1)
    nx.draw_networkx_labels(G, pos=pos, font_size=8)
    nx.draw_networkx_edges(G, pos=pos, edge_color='k', alpha=0.8)
    plt.savefig('network_{}.pdf'.format(separate), bbox_inches='tight', transparent=True, pad_inches=0.1)
    plt.show()

if __name__ == '__main__':
    # make_instance()
    # main()
    # write_res_analysis_csv()
    # num_boxplot()
    # obj_boxplot()
    # scatter_ins()
    draw_route()
    # google_map_draw()

    # for name, res_li in read_res(0):
    #     print(name)
    #     temp = 0
    #     temp1 = 0
    #     for res in res_li:
    #         temp += 1
    #         try:
    #             a = [key for key in res[2]]
    #             if '5' in a:
    #                 temp1 += 1
    #         except:
    #             pass
    #     print(temp1 / temp)

# img = mpimg.imread('s_picture_satellite.png')
# print(img[0][1])
# plt.imshow(img, alpha=0.9)
