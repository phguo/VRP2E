import os
import json
import pandas as pd
import csv


def read_res():
    path = './res_stand/'
    files = os.listdir(path)
    files = sorted(files, key=lambda file_name: file_name[9:] + file_name[6:8])
    files = [name for name in files if not name[0] == '.']

    for file in files:
        file_path = path + file
        with open(file_path, 'r') as f:
            res = json.load(f)
            yield (file, res)

def write_res_analysis_csv():
    headers = ['instance_name','non_dominated_solution_num','time_consuming',
              'min_obj1','mean_obj1','median_obj1','max_obj1',
              'min_obj2','mean_obj2','median_obj2','max_obj2',
              'min_obj3','mean_obj3','median_obj3','max_obj3',
              'evaluation1', 'evaluation2', 'evaluation3']
    rows = []
    for ins_name, res_li in read_res():
        if int(ins_name[6:8]) % 2 == 0:
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

    with open('res_analysis.csv','w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)
