import os
from matplotlib import pyplot as plt
import copy
import random
import json


random.seed(1)

def read_data():
    # single depot & single production instance
    path = './data/2evrp_instances_unified_breunig/'
    files = os.listdir(path)
    for file in files:
        file_path = path + file
        instance = {'name': file.strip('.dat'),
                    'depot': [], 'satellite': [], 'customer': [],
                    'vehicle1_num': None, 'vehicle2_num': None,
                    'vehicle1_cap': None, 'vehicle2_cap': None,
                    'satellite_cap': None}
        with open(file_path,'r') as f:
            temp_li = []
            for line in f:
                line = line.strip('\n')
                temp_li.append(line)
            # vehicle
            instance['vehicle1_num'], instance['vehicle1_cap'] = [int(a) for a in temp_li[2].split(',')[:2]]
            instance['vehicle2_num'], instance['vehicle2_cap'] = [int(a) for a in temp_li[5].split(',')[1:3]]
            depot_satellite = temp_li[8].split('   ')
            depot_x, depot_y = depot_satellite[0].split(',')
            # depot
            instance['depot'].append([(int(depot_x), int(depot_y)), -1])
            for x_y in depot_satellite[1:]:
                satellite_x, satellite_y = x_y.split(',')
            # satellite
                instance['satellite'].append([(int(satellite_x), int(satellite_y)), -1])
            for x_y_d in temp_li[11].split('   '):
                customer_x, customer_y, customer_d = x_y_d.split(',')
            # customer
                instance['customer'].append([(int(customer_x), int(customer_y)), [int(customer_d)]])
        yield(instance)

def draw_chosen():  # draw the chosen instance: set4 & set5
    instance_dic = {}
    for instance in read_data():
        instance_dic[instance['name']] = instance
    draw_li = []
    for key in instance_dic:
        if ('Set4' in key) and (not 'b' in key):
            draw_li.append(key)
    for ins_name in draw_li:
        satellite_num = len(instance_dic[ins_name]['satellite'])
        customer_num = len(instance_dic[ins_name]['customer'])
        print(ins_name, satellite_num, customer_num)
        print(ins_name)
        instance = instance_dic[ins_name]
        for key in ['depot', 'satellite', 'customer']:
            x_li, y_li = [], []
            for a in instance[key]:
                x_li.append(a[0][0])
                y_li.append(a[0][1])
            plt.title("{}".format(instance['name']))
            plt.scatter(x_li, y_li, alpha=0.5)
        plt.savefig("{}.pdf".format(ins_name), transparent=True, bbox_inches='tight', pad_inches=0.1)
        plt.show()

def make_experiment_data():
    instance_dic = {}
    for instance in read_data():
        instance_dic[instance['name']] = instance

    set4_dic = {key: instance_dic[key] for key in instance_dic if 'Set4a' in key}
    chosen_ins_name = ['Set4a_Instance50-%s' % i for i in range(1, 19)]  # 1~18
    # 1,2,30 & 1,2,50  <-- 1,2,50
    for ins in chosen_ins_name:
        # add supply 1
        demand_li = [i[1][0] for i in set4_dic[ins]['customer']]
        set4_dic[ins]['depot'][0][1] = 0.8 * sum(demand_li)  # FIXME

    chosen_ins_name = ['Set4a_Instance50-%s' % i for i in range(19, 37)]  # 19~36
    # 2,3,30 & 2,3,50  <-- 1,3,50
    for ins in chosen_ins_name:
        # add depot
        set4_dic[ins]['depot'].append([(3000, 16000), -1])

        # add demand 1
        demand_li = [i[1][0] for i in set4_dic[ins]['customer']]
        lower, upper = min(demand_li), max(demand_li)
        for cus in set4_dic[ins]['customer']:
            cus[1].append(random.randrange(int(0.5 * lower), int(0.5 * upper)))
        demand_li2 = [i[1][1] for i in set4_dic[ins]['customer']]

        # add supply 2
        set4_dic[ins]['depot'][0][1] = 0.8 * sum(demand_li)  # FIXME
        set4_dic[ins]['depot'][1][1] = 0.8 * sum(demand_li2)  # FIXME


    chosen_ins_name = ['Set4a_Instance50-%s' % i for i in range(37, 55)]  # 37~54
    # 3,5,30 & 3,5,50  <-- 1,5,50
    for ins in chosen_ins_name:
        # add depot
        set4_dic[ins]['depot'].append([(3000, 16000), -1])
        set4_dic[ins]['depot'].append([(12000, 5000), -1])

        # add demand 2
        demand_li = [i[1][0] for i in set4_dic[ins]['customer']]
        lower, upper = min(demand_li), max(demand_li)
        for cus in set4_dic[ins]['customer']:
            cus[1].append(random.randrange(int(0.5 * lower), int(0.5 * upper)))
            cus[1].append(random.randrange(int(0.3 * lower), int(0.3 * upper)))
        demand_li2 = [i[1][1] for i in set4_dic[ins]['customer']]
        demand_li3 = [i[1][2] for i in set4_dic[ins]['customer']]

        # add supply 3
        set4_dic[ins]['depot'][0][1] = 0.8 * sum(demand_li)  # FIXME
        set4_dic[ins]['depot'][1][1] = 0.8 * sum(demand_li2)  # FIXME
        set4_dic[ins]['depot'][2][1] = 0.8 * sum(demand_li3)  # FIXME

    # 30 customer
    set4_dic_30 = copy.deepcopy(set4_dic)
    for ins in set4_dic_30:
        set4_dic_30[ins]['customer'] = random.sample(set4_dic_30[ins]['customer'], 30)
    # FIXME supply_amount * 3/5
    for ins in set4_dic_30:
        try:
            demand_li = [i[1][0] for i in set4_dic_30[ins]['customer']]
            set4_dic_30[ins]['depot'][0][1] = 0.8 * sum(demand_li)
        except: pass
        try:
            demand_li2 = [i[1][1] for i in set4_dic_30[ins]['customer']]
            set4_dic_30[ins]['depot'][1][1] = 0.8 * sum(demand_li2)
        except: pass
        try:
            demand_li3 = [i[1][2] for i in set4_dic_30[ins]['customer']]
            set4_dic_30[ins]['depot'][2][1] = 0.8 * sum(demand_li3)
        except: pass



    return(set4_dic, set4_dic_30)

def write_json_instance():
    dic_50, dic_30 = make_experiment_data()
    chosen_ins_name = ['Set4a_Instance50-%s' % i for i in range(1, 55)]
    for dic in [dic_50, dic_30]:
        for ins in chosen_ins_name:
            depot = dic[ins]['depot']
            satellite = dic[ins]['satellite']
            customer = dic[ins]['customer']
            vehicle1_num,  vehicle2_num= dic[ins]['vehicle1_num'], dic[ins]['vehicle2_num']
            vehicle1_cap, vehicle2_cap = dic[ins]['vehicle1_cap'], dic[ins]['vehicle2_cap']
            name = '{}_{}-{}-{}'.format(ins.replace('Instance50-', ''), len(depot), len(satellite), len(customer))
            if len(name) == 14:
                name = name[:6] + '0' + name[6:]

            format_depot = {i:(depot[i][0], depot[i][1]) for i in range(len(depot))}
            format_satellite = {i+len(depot):(satellite[i][0], float('inf')) for i in range(len(satellite))}
            format_customer = {i+len(depot)+len(satellite):(customer[i][0], customer[i][1]) for i in range(len(customer))}

            INSTANCE = {'name': name,
                        'depot': format_depot, 'satellite': format_satellite, 'customer': format_customer,
                        'vehicle1_num': vehicle1_num * len(depot) * 1, 'vehicle2_num': vehicle2_num * len(depot) * 1,  # FIXME
                        'vehicle1_cap': vehicle1_cap, 'vehicle2_cap': vehicle2_cap,
                        'satellite_cap': float("inf")}

            json_data = json.dumps(INSTANCE, sort_keys=True, indent=2, separators=(',', ':'))
            with open('./test_data/{}.json'.format(name), 'wt') as f:
                f.write(json_data)

def load_instance_json():
    path = './test_data/'
    files = os.listdir(path)
    files = [name for name in files if not name[0] == '.']
    files = sorted(files, key=lambda file_name: file_name[9:] + file_name[6:8])
    for file in files:
        file_path = path + file
        with open(file_path, 'r') as f:
            instance = json.load(f)
            for s in ['depot', 'satellite', 'customer']:
                instance[s] = {int(key): instance[s][key] for key in instance[s]}
            yield (instance)

def draw():
    for ins in load_instance_json():
        name = ins['name']
        if name in ['Set4a_44_3-5-50', 'Set4a_52_3-5-50']:
            print(ins)
            for key in ['depot', 'satellite', 'customer']:
                x_li, y_li = [], []
                for num in ins[key]:
                    x_li.append(ins[key][num][0][0])
                    y_li.append(ins[key][num][0][1])
                plt.title("{}".format(ins['name']))
                plt.scatter(x_li, y_li, alpha=0.5)
            plt.savefig("{}.pdf".format(name), transparent=True, bbox_inches='tight', pad_inches=0.1)
            plt.show()

if __name__ == '__main__':
    for ins in load_instance_json():
        if int(ins['name'][6:8]) % 2 == 0:
            ins_name = ins['name']
            supply = [ins['depot'][key][1] for key in ins['depot']]
            vehicle1_num, vehicle2_num = ins['vehicle1_num'], ins['vehicle2_num']
            vehicle1_cap, vehicle2_cap = ins['vehicle1_cap'], ins['vehicle2_cap']
            supply1 = supply[0]
            try: supply2 = supply[1]
            except: supply2 = 0
            try: supply3 = supply[2]
            except: supply3 = 0
            print('{},{},{},{},{},{},{},{}'.format(ins_name, str(vehicle1_num), str(vehicle1_cap), str(vehicle2_num), str(vehicle2_cap),
                                                   str(supply1), str(supply2), str(supply3)))