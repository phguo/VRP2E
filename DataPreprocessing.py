import os
import LRP2E
from matplotlib import pyplot as plt
from collections import OrderedDict

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



def add_depot(instance):
    # add depot & production
    new_instance = {}
    return(new_instance)



