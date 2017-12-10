import random
from matplotlib import pyplot as plt
import numpy as np
import math
import time
import itertools
from collections import OrderedDict
import profile
import copy


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
        self.pop_size = 5
        self.f = 0.05
        self.mutt_prob = 0.05
        self.violation_weigh = 0.5

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

    def depot_satellite_rout_(self, assignment):
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

    def depot_satellite_rout(self, assignment):
        # route depot -> satellite is generated according to the assignment.
        spa = self.satellite_production_amount(assignment)
        satellite_li = []
        for key in assignment:
            s = assignment[key][0]
            if s not in satellite_li:
                satellite_li.append(s)

        depot_satellite_rout = {depot:[[]] for depot in self.depot}
        for depot in self.depot:
            temp_cap = 0
            for satellite in satellite_li:
                temp_cap += spa[satellite][depot]
                if temp_cap < self.vehicle1_cap:
                    depot_satellite_rout[depot][-1].append(satellite)
                else:
                    temp_cap = spa[satellite][depot]
                    depot_satellite_rout[depot].append([])
                    depot_satellite_rout[depot][-1].append(satellite)

        return (depot_satellite_rout)

    def satellite_customer_rout_(self, assignment):
        # route satellite -> customer is generated according to the assignment.
        cpt = self.customer_production_total(assignment)

        satellite_customer_assignment = {stl: [] for stl in self.satellite}
        # satellite_customer_assignment = {assignment[key][0] for key in assignment}
        for stl in self.satellite:
            for key in assignment:
                if stl == assignment[key][0]:
                    satellite_customer_assignment[stl].append(key)
        # print(satellite_customer_assignment)
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
        # print(satellite_customer_rout)
        return(satellite_customer_rout)

    def satellite_customer_rout(self, assignment):
        # route satellite -> customer is generated according to the assignment.
        cpt = self.customer_production_total(assignment)

        satellite_customer_assignment = {assignment[key][0]:[] for key in assignment}
        for stl in self.satellite:
            for key in assignment:
                if stl == assignment[key][0]:
                    satellite_customer_assignment[stl].append(key)

        satellite_customer_rout = {satellite:[[]] for satellite in satellite_customer_assignment}
        for stl in satellite_customer_rout:
            temp_cap = 0
            for cst in satellite_customer_assignment[stl]:
                temp_cap += cpt[cst]
                if temp_cap < self.vehicle2_cap:
                    satellite_customer_rout[stl][-1].append(cst)
                else:
                    temp_cap = cpt[cst]
                    satellite_customer_rout[stl].append([])
                    satellite_customer_rout[stl][-1].append(cst)
        return(satellite_customer_rout)

    def rand_ind(self):
        # the total amount of every production is separated to some parts randomly.
        customer_li = [key for key in self.customer]
        random.shuffle(customer_li)

        assignment = OrderedDict({customer: [satellite]
                                  for customer, satellite
                                  in zip(customer_li,
                                  [random.choice([key for key in self.satellite]) for _ in range(len(self.customer))])})
        # print(assignment)
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

    def obj_time_(self, ind):
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

    def obj_time(self, ind):
        obj_value = 0
        d_s, s_c = ind[1], ind[2]
        assert(set(d_s) & set(s_c) == set())
        temp_dic = copy.deepcopy(d_s)
        temp_dic.update(s_c)

        for key in temp_dic:
            for li in temp_dic[key]:
                temp_li = [key] + li + [key]
                for i in range(len(temp_li)):
                    try:
                        obj_value += math.sqrt(
                            (self.loc[li[i]][0] - self.loc[li[i + 1]][0]) ** 2
                          + (self.loc[li[i]][1] - self.loc[li[i + 1]][1]) ** 2)
                    except: pass
        return(obj_value)

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
        return(-np.sum(temp_li), -np.var(temp_li))

    def crossover(self, ind0, ind1):
        assignment0, assignment1 = copy.deepcopy(ind0[0]), copy.deepcopy(ind1[0])
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
        # new_assignment = OrderedDict({key: ind_assignment[key][:] for key in ind_assignment})
        new_assignment = copy.deepcopy(ind_assignment)

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

    def not_feasible(self, ind):
        assignment = ind[0]

        # satellite violation value: production amount exceed satellite capacity.

        # customer violation value: production amount exceed customer need or production amount is negative.

        # vehicle number violation value:

        # depot violation value

        violation_value = 0
        return(violation_value)

    def constraint_choose(self, obj_func, ind0, ind1):
        if not self.not_feasible(ind0) and not self.not_feasible(ind1):
            if obj_func(ind0) < obj_func(ind1):
                chosen_one = ind0[:]
            else:
                chosen_one = ind1[:]
        elif self.not_feasible(ind0) and not self.not_feasible(ind1):
            chosen_one = ind1[:]
        elif not self.not_feasible(ind0) and self.not_feasible(ind1):
            chosen_one = ind0[:]
        else:
            if obj_func(ind0) + self.violation_weigh * self.not_feasible(ind0) \
             < obj_func(ind1) + self.violation_weigh * self.not_feasible(ind1):
                chosen_one = ind0[:]
            else:
                chosen_one = ind1[:]
        return(chosen_one)

    def single_objective_select(self, obj_func, ind1, ind2, pop):
        if random.random() < self.mutt_prob:
            temp_ind1, temp_ind2 = self.mutation(ind1, pop), self.mutation(ind2, pop)
        else:
            temp_ind1, temp_ind2 = ind1[:], ind2[:]
        temp_pair = self.crossover(temp_ind1, temp_ind2)
        offspring = []
        offspring.append(self.constraint_choose(obj_func, temp_pair[0], ind1))
        offspring.append(self.constraint_choose(obj_func, temp_pair[1], ind2))
        return(offspring)

    def single_objective_evolution(self, obj_func, pop):
        # input: population & evaluate function
        # output: the best feasible individual & offspring population
        offspring_population = []
        for pair in itertools.combinations(pop, 2):
            for a in self.single_objective_select(obj_func, pair[0], pair[1], pop):
                offspring_population.append(a)
        return(offspring_population)

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

    for ind in pop:
        assign, depot_satellite_rout, satellite_customer_rout = ind[0], ind[1], ind[2]
        print(['%-2s' % a for a in assign])
        print(['%-2s' % assign[a][0] for a in assign])
        print(depot_satellite_rout)
        print(v.depot_satellite_rout_(assign))
        print(satellite_customer_rout)
        print(v.satellite_customer_rout_(assign))
        print(v.obj_time(ind))
        print('\n')
    print('===' * 50)
    new_pop = v.single_objective_evolution(v.obj_time, pop)
    for ind in new_pop:
        assign, depot_satellite_rout, satellite_customer_rout = ind[0], ind[1], ind[2]
        print(['%-2s' % a for a in assign])
        print(['%-2s' % assign[a][0] for a in assign])
        print(depot_satellite_rout)
        print(satellite_customer_rout)
        print(v.obj_time(ind))
        print('\n')



if __name__ == '__main__':
    main()
    # profile.run('main()')