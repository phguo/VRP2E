import os
import LRP2E
from matplotlib import pyplot as plt

def read_data():
    # original data is single depot & single production.
    path = './data/2evrp_instances_unified_breunig/'
    files = os.listdir(path)
    for file in files:
        file_path = path + file
        instance = {'name': file,
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

def add_depot(instance):
    new_instance = {}
    return(new_instance)

PARAMETERS = {'pop_size': 500, 'offspring_size': 30, 'archive_size': 300,
              'obj_num': 3, 'f': 0.05,
              'mutt_prob': 0.05, 'cross_prob': 0.5,
              'violation_weigh': 0.5,
              'not_feasible_weigh': {'depot':0.2, 'satellite':0.2, 'customer':0.2, 'vehicle':0.4}}

instance_li = []
for instance in read_data():
    instance_li.append(instance)
instance_li.sort(key=lambda x:len(x['satellite']), reverse=1)



for instance in instance_li:
    for key in ['depot', 'satellite', 'customer']:
        x_li, y_li = [], []
        for a in instance[key]:
            x_li.append(a[0][0])
            y_li.append(a[0][1])
        plt.title(instance['name'])
        plt.scatter(x_li, y_li)
    plt.show()
    break



