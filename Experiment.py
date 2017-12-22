import LRP2E
import DataPreprocessing
import itertools
import time
import requests
import json
import traceback


PARAMETERS = {'f': 0.5,
              'pop_size': 500, 'offspring_size': 300, 'archive_size': 400, 'k': 300, 'mutt_prob': 0.3, 'iter_times': 100}

def classify_ins():
    ins_li = [[],[],[]]
    for ins in DataPreprocessing.load_instance_json():
        name = ins['name']
        ins_id = int(name[6:8])
        if ins_id in [i for i in range(1,19)]:
            ins_li[0].append(ins)
        elif int(ins_id) in [i for i in range(19,37)]:
            ins_li[1].append(ins)
        elif int(ins_id) in [i for i in range(37,55)]:
            ins_li[2].append((ins))
    return(ins_li)

def parameter_calibration():
    f1 = [i/10 for i in range(3, 8)]
    f2, f3 = f1[:], f1[:]
    violation_weigh = 1
    depot_violation = []
    satellite_violation = 0
    customer_violation = []
    vehicle_violation = []

    pop_size = []
    offspring_size = []
    archive_size = []
    k = []
    mutt_prob = []
    iter_times = 1000
    parameters = {}
    return(parameters)

def run(ins):
    print('==' * 40)
    print('Solving the instance:', ins['name'])
    t1 = time.clock()
    res = LRP2E.main(ins, PARAMETERS)
    t2 = time.clock()
    for ind in res:
        print('-' * 40)
        print(ind[1])
        print(ind[2])
        print(ind[3])
        print(ind[4])
        print(ind[5])
#     # res.append(t2 - t1)
#     print('time consuming:', t2 - t1)
#     json_data = json.dumps(res, sort_keys=True, indent=2, separators=(',', ':'))
#     with open('./res/{}.json'.format(ins['name']), 'wt') as f:
#         f.write(json_data)
#
# def sc_send(title, content='', key='SCU11157Ta6c223fc34f1d1a2936565187b49b89359a2615784ee4'):
#     url = 'http://sc.ftqq.com/' + key + '.send?text=' + title + '&desp=' + content
#     r = requests.get(url)
#     if r.status_code == 200:
#         return ('OK')
#     else:
#         return ('Opps')
#
# ins_name = 'BUG'
# try:
#     t1 = time.clock()
#     for ins in classify_ins()[0]:
#         ins_name = ins['name']
#         run(ins)
#     t2 = time.clock()
#     sc_send('第一类测试用例运行完毕', str((t2 - t1)/60) + '分钟')
#
#     t1 = time.clock()
#     for ins in classify_ins()[1]:
#         ins_name = ins['name']
#         run(ins)
#     t2 = time.clock()
#     sc_send('第二类测试用例运行完毕', str((t2 - t1)/60) + '分钟')
#
#     t1 = time.clock()
#     for ins in classify_ins()[2]:
#         ins_name = ins['name']
#         run(ins)
#     t2 = time.clock()
#     sc_send('第三类测试用例运行完毕', str((t2 - t1)/60) + '分钟')
#
# except:
#     title = ins_name
#     content = traceback.format_exc()
#     sc_send(title, content)
for ins in classify_ins()[2][7:]:
    print(ins['name'])
    run(ins)
    break