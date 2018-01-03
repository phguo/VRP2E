import LRP2E
import DataPreprocessing
import time
import requests
import json
import traceback
import key


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


def run(ins):
    PARAMETERS = {'f': 0.5,
                  'pop_size': 500, 'offspring_size': 300, 'archive_size': 400, 'k': 300, 'mutt_prob': 0.3,
                  'iter_times': 100}
    print('==' * 40)
    print('Solving the instance:', ins['name'])
    t1 = time.clock()
    res = LRP2E.main(ins, PARAMETERS, 1)
    t2 = time.clock()
    res.append(t2 - t1)
    print('time consuming:', t2 - t1)
    json_data = json.dumps(res, sort_keys=True, indent=2, separators=(',', ':'))
    with open('./res_stand_separate/{}.json'.format(ins['name']), 'wt') as f:
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
        t1 = time.clock()
        for ins in classify_ins()[0]:
            ins_name = ins['name']
            run(ins)
        t2 = time.clock()
        sc_send('第一类测试用例运行完毕', str((t2 - t1)/60) + '分钟')

        t1 = time.clock()
        for ins in classify_ins()[1]:
            ins_name = ins['name']
            run(ins)
        t2 = time.clock()
        sc_send('第二类测试用例运行完毕', str((t2 - t1)/60) + '分钟')

        t1 = time.clock()
        for ins in classify_ins()[2]:
            ins_name = ins['name']
            run(ins)
        t2 = time.clock()
        sc_send('第三类测试用例运行完毕', str((t2 - t1)/60) + '分钟')

    except:
        title = ins_name
        content = traceback.format_exc()
        sc_send(title, content)
