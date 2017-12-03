import random
from matplotlib import pyplot as plt
import numpy as np
import math
import time
import itertools
from collections import OrderedDict


random.seed(1)

DEPOT_NUM, SATELLITE_NUM, CUSTOMER_NUM = 3, 5, 9
SATELLITE_CAP, VEHICLE1_CAP, VEHICLE2_CAP = 500, 60, 60

DEPOT     = {i:((random.uniform(0, 10), random.uniform(0, 10)), 100)
             for i in range(DEPOT_NUM)}
SATELLITE = {i:((random.uniform(0, 10), random.uniform(0, 10)), 1000)
             for i in range(max(DEPOT) + 1, max(DEPOT) + 1 + SATELLITE_NUM)}
CUSTOMER  = {i:((random.uniform(0, 10), random.uniform(0, 10)), [20] * DEPOT_NUM)
             for i in range(max(SATELLITE) + 1, max(SATELLITE) + 1 + CUSTOMER_NUM)}


class VRP2E:
    def __init__(self):
        # DATA
        self.depot = DEPOT
        self.satellite = SATELLITE
        self.customer = CUSTOMER
        self.satellite_cap = SATELLITE_CAP
        self.vehicle1_cap = VEHICLE1_CAP
        self.vehicle2_cap = VEHICLE2_CAP
        self.loc = {}
        for data in [self.depot, self.satellite, self.customer]:
            for i in data:
                self.loc[i] = data[i][0]

        # evolutionary algorithm parameter
        self.pop_size = 1000
        self.f = 0.05
        self.mutt_prob = 0.05

    def satellite_production_amount(self, assignment):
        result_dic = {i: [0] * len(self.depot) for i in self.satellite}
        for i in self.satellite:
            for key in assignment:
                if i == assignment[key][0]:
                    for p in range(len(self.depot)):
                        result_dic[i][p] += assignment[key][p + 1]
        return(result_dic)

    def customer_production_total(self, assignment):
        result_dic = {i: np.sum(assignment[i][1:]) for i in self.customer}
        return(result_dic)

    def depot_satellite_rout(self, assignment):
        # route depot -> satellite is generated according to the assignment.
        spa = self.satellite_production_amount(assignment)
        satellite_li = []
        for key in assignment:
            s = assignment[key][0]
            if s not in satellite_li:
                satellite_li.append(s)

        depot_satellite_rout = []
        for depot in self.depot:
            depot_satellite_rout.append('/')
            # random.shuffle(satellite_li)  # vehicle from depot to satellites
            depot_satellite_rout.append(depot)
            temp_cap = 0
            for satellite in satellite_li:
                temp_cap += spa[satellite][depot]
                if temp_cap < self.vehicle1_cap:
                    depot_satellite_rout.append(satellite)
                else:
                    temp_cap = spa[satellite][depot]
                    depot_satellite_rout.append(depot)  # ('*')
                    depot_satellite_rout.append(satellite)
        return(depot_satellite_rout)

    def satellite_customer_rout(self, assignment):
        # route satellite -> customer is generated according to the assignment.
        cpt = self.customer_production_total(assignment)

        satellite_customer_assignment = {stl: [] for stl in self.satellite}
        for stl in self.satellite:
            for key in assignment:
                if stl == assignment[key][0]:
                    satellite_customer_assignment[stl].append(key)

        satellite_customer_rout = []
        for stl in self.satellite:
            satellite_customer_rout.append('/')
            satellite_customer_rout.append(stl)
            temp_cap = 0
            for cst in satellite_customer_assignment[stl]:
                temp_cap += cpt[cst]
                if temp_cap < self.vehicle2_cap:
                    satellite_customer_rout.append(cst)
                else:
                    temp_cap = cpt[cst]
                    satellite_customer_rout.append(stl)  # ('*')
                    satellite_customer_rout.append(cst)
        return(satellite_customer_rout)

    def rand_ind(self):
        # the total amount of every production is separated to some parts randomly.
        customer_li = [key for key in self.customer]
        random.shuffle(customer_li)

        assignment = OrderedDict({customer: [satellite]
                                  for customer, satellite
                                  in zip(customer_li,
                                  [random.choice([key for key in self.satellite]) for _ in range(len(self.customer))])})
        for key in assignment:
            for i in range(len(self.depot)):
                assignment[key].append(random.randrange(0,20))

        d_s = self.depot_satellite_rout(assignment)
        s_c = self.satellite_customer_rout(assignment)
        individual = [assignment, d_s, s_c]
        return(individual)

    def greedy_ind(self):
        individual = []
        return(individual)

    def rand_pop(self):
        return([self.rand_ind() for _ in range(self.pop_size)])

    def obj_time(self, ind):
        # sum(ti, for all the i)
        # customer waiting time || total vehicle traveling time ?
        obj_value_1 = 0
        d_s = ind[1]
        s_c = ind[2]
        for li in [d_s, s_c]:
            for i in range(len(li)):  # return time cost is not considered
                if i < len(li) - 1 and li[i] != '/' and li[i + 1] != '/':
                    obj_value_1 += math.sqrt(
                          (self.loc[li[i]][0] - self.loc[li[i + 1]][0]) ** 2
                        + (self.loc[li[i]][1] - self.loc[li[i + 1]][1]) ** 2)
        return(obj_value_1)

    def customer_satisfaction(self, ind):
        assign = ind[0]
        customer_satisfaction_dic = {i: [0] * len(self.depot) for i in self.customer}

        for key in assign:
            for j in range(len(self.depot)):
                customer_satisfaction_dic[key][j] = assign[key][j+1] / self.customer[key][1][j]

        customer_satisfaction_dic = {i:[customer_satisfaction_dic[i], sum(customer_satisfaction_dic[i])]
                                     for i in customer_satisfaction_dic}  # different production share same weight
        return(customer_satisfaction_dic)

    def obj_satisfaction_equity(self, ind):
        customer_satisfaction_dic = self.customer_satisfaction(ind)
        temp_li = [customer_satisfaction_dic[i][1] for i in customer_satisfaction_dic]
        return(np.sum(temp_li), np.var(temp_li))

    def crossover(self, ind0, ind1):
        assignment0, assignment1 = ind0[0], ind1[0]
        customer_order = [key for key in assignment0]
        cross_start, cross_end = sorted([random.randint(0, len(customer_order) - 1),
                                         random.randint(0, len(customer_order) - 1)])
        cross_points = customer_order[cross_start:cross_end + 1]
        for point in cross_points:
            assignment0[point], assignment1[point] = assignment1[point], assignment0[point]

        depot_satellite_rout_0 = self.depot_satellite_rout(assignment0)
        depot_satellite_rout_1 = self.depot_satellite_rout(assignment1)
        satellite_customer_rout_0 = self.satellite_customer_rout(assignment0)
        satellite_customer_rout_1 = self.satellite_customer_rout(assignment1)
        ind0_son = [assignment0, depot_satellite_rout_0, satellite_customer_rout_0]
        ind1_son = [assignment1, depot_satellite_rout_1, satellite_customer_rout_1]
        return(ind0_son, ind1_son)

    def mutation(self, ind, pop, pop_best=[], archive_best=[], f=0.05):
        pop_best, archive_best = random.choice(pop), random.choice(pop)  # just for test

        ind1_assignment, ind2_assignment = random.choice(pop)[0], random.choice(pop)[0]
        pop_best_assignment, archive_best_assignment = pop_best[0], archive_best[0]
        ind_assignment = ind[0]
        new_assignment = OrderedDict({key: ind_assignment[key][:] for key in ind_assignment})

        # mutation of delivery amount --> follow the method of Wang(2016)(8)
        for key in new_assignment:
            for i in range(1, len(new_assignment[key])):
                new_assignment[key][i] = ind_assignment[key][i] \
                                         + f * (pop_best_assignment[key][i] - ind_assignment[key][i]) \
                                         + f * (ind1_assignment[key][i] - ind2_assignment[key][i]) \
                                         + f * (archive_best_assignment[key][i] - ind_assignment[key][i])

        # mutation of rout structure --> reverse the satellite order in assignment chromosome
        satellite_order = [new_assignment[key][0] for key in new_assignment]
        for key in new_assignment:
            new_assignment[key][0] = satellite_order[-1]
            satellite_order.pop()

        depot_satellite_rout = self.depot_satellite_rout(new_assignment)
        satellite_customer_rout = self.satellite_customer_rout(new_assignment)
        new_ind = [new_assignment, depot_satellite_rout, satellite_customer_rout]
        return(new_ind)

    def single_objective_evolution(self, func, pop):
        # input: population & evaluate function
        # output: the best feasible individual & offspring population
        return()

    def multi_objective_evolution(self):

        return()


def timer(func):
    def wrapTheFunction():
        start_time = time.clock()
        func()
        end_time = time.clock()
        print('time consuming:', end_time - start_time)
    return wrapTheFunction

@timer
def main():
    v = VRP2E()
    pop = v.rand_pop()
    for pair in itertools.combinations(pop, 1):
        v.mutation(pair[0], pop)


if __name__ == '__main__':
    main()