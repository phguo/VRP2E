import os
import json

def read_res():
    path = './res/'
    files = os.listdir(path)
    files = sorted(files)
    files = [name for name in files if not name[0] == '.']

    for file in files:
        file_path = path + file
        with open(file_path, 'r') as f:
            res = json.load(f)
            yield (file, res)

for ins_name, res_li in read_res():
    for res in res_li:
        print(res[-2])
    print('-' * 30)