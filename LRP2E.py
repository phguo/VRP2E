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
# TODO use the downloaded data.
DEPOT_NUM, SATELLITE_NUM, CUSTOMER_NUM = 3, 5, 9
SATELLITE_CAP, VEHICLE1_CAP, VEHICLE2_CAP = 500, 60, 60
VEHICLE1_NUM, VEHICLE2_NUM = 10, 8

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
        self.vehicle1_cap, self.vehicle2_cap = VEHICLE1_CAP, VEHICLE2_CAP
        self.vehicle1_num, self.vehicle2_num = VEHICLE1_NUM, VEHICLE2_NUM
        self.loc = {}
        for data in [self.depot, self.satellite, self.customer]:
            for i in data:
                self.loc[i] = data[i][0]

        # evolutionary algorithm parameter
        self.pop_size = 500
        self.offspring_size = 30
        self.archive_size = 300
        self.obj_num = 3
        self.k = self.pop_size
        self.f = 0.05
        self.mutt_prob = 0.05
        self.cross_prob = 0.5
        self.violation_weigh = 0.5
        self.not_feasible_weigh = {'depot':0.2, 'satellite':0.2, 'customer':0.2, 'vehicle':0.4}

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

    def depot_satellite_route(self, assignment):
        # route depot -> satellite is generated according to the assignment.
        spa = self.satellite_production_amount(assignment)
        satellite_li = []
        for key in assignment:
            s = assignment[key][0]
            if s not in satellite_li:
                satellite_li.append(s)

        depot_satellite_route = {depot:[[]] for depot in self.depot}
        for depot in self.depot:
            temp_cap = 0
            for satellite in satellite_li:
                temp_cap += spa[satellite][depot]
                if temp_cap < self.vehicle1_cap:
                    depot_satellite_route[depot][-1].append(satellite)
                else:
                    temp_cap = spa[satellite][depot]
                    depot_satellite_route[depot].append([])
                    depot_satellite_route[depot][-1].append(satellite)

        return (depot_satellite_route)

    def satellite_customer_route(self, assignment):
        # route satellite -> customer is generated according to the assignment.
        cpt = self.customer_production_total(assignment)

        satellite_customer_assignment = {assignment[key][0]:[] for key in assignment}
        for stl in self.satellite:
            for key in assignment:
                if stl == assignment[key][0]:
                    satellite_customer_assignment[stl].append(key)

        satellite_customer_route = {satellite:[[]] for satellite in satellite_customer_assignment}
        for stl in satellite_customer_route:
            temp_cap = 0
            for cst in satellite_customer_assignment[stl]:
                temp_cap += cpt[cst]
                if temp_cap < self.vehicle2_cap:
                    satellite_customer_route[stl][-1].append(cst)
                else:
                    temp_cap = cpt[cst]
                    satellite_customer_route[stl].append([])
                    satellite_customer_route[stl][-1].append(cst)
        return(satellite_customer_route)

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

        d_s = self.depot_satellite_route(assignment)
        s_c = self.satellite_customer_route(assignment)
        individual = [assignment, d_s, s_c]
        individual += [[self.obj_time(individual),
                       self.obj_satisfaction_equity(individual)[0],
                       self.obj_satisfaction_equity(individual)[1]]]
        return(individual)

    def greedy_ind(self):
        individual = []
        return(individual)

    def rand_pop(self):
        return([self.rand_ind() for _ in range(self.pop_size)])

    def obj_time(self, ind):
        obj_value = 0
        d_s, s_c = ind[1], ind[2]
        assert(set(d_s) & set(s_c) == set())
        temp_dic = copy.deepcopy(d_s)
        temp_dic.update(s_c)

        for key in temp_dic:
            for li in temp_dic[key]:
                temp_li = [key] + li + [key]
                for i in range(len(temp_li) - 1):
                    obj_value += math.sqrt(
                        (self.loc[temp_li[i]][0] - self.loc[temp_li[i + 1]][0]) ** 2
                      + (self.loc[temp_li[i]][1] - self.loc[temp_li[i + 1]][1]) ** 2)
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
        return(-np.sum(temp_li), np.var(temp_li))

    def crossover(self, ind0, ind1):
        assignment0, assignment1 = copy.deepcopy(ind0[0]), copy.deepcopy(ind1[0])
        customer_order = [key for key in assignment0]
        cross_start, cross_end = sorted([random.randint(0, len(customer_order) - 1),
                                         random.randint(0, len(customer_order) - 1)])
        cross_points = customer_order[cross_start:cross_end + 1]
        for point in cross_points:
            assignment0[point], assignment1[point] = assignment1[point], assignment0[point]

        depot_satellite_route_0 = self.depot_satellite_route(assignment0)
        depot_satellite_route_1 = self.depot_satellite_route(assignment1)
        satellite_customer_route_0 = self.satellite_customer_route(assignment0)
        satellite_customer_route_1 = self.satellite_customer_route(assignment1)
        ind0_son = [assignment0, depot_satellite_route_0, satellite_customer_route_0]
        ind1_son = [assignment1, depot_satellite_route_1, satellite_customer_route_1]
        ind0_son += [[self.obj_time(ind0_son), self.obj_satisfaction_equity(ind0_son)[0],
                     self.obj_satisfaction_equity(ind0_son)[1]]]
        ind1_son += [[self.obj_time(ind1_son), self.obj_satisfaction_equity(ind1_son)[0],
                     self.obj_satisfaction_equity(ind1_son)[1]]]
        return(ind0_son, ind1_son)

    def mutation(self, ind, pop, pop_best=[], archive_best=[], f=0.05):
        pop_best, archive_best = pop[0], random.choice(pop)  # just for test

        ind1_assignment, ind2_assignment = random.choice(pop)[0], random.choice(pop)[0]
        pop_best_assignment, archive_best_assignment = pop_best[0], archive_best[0]
        ind_assignment = ind[0]
        # new_assignment = OrderedDict({key: ind_assignment[key][:] for key in ind_assignment})
        new_assignment = copy.deepcopy(ind_assignment)

        # mutation of delivery amount --> follow the method of Wang(2016)(8)
        # TODO parameter ?
        for key in new_assignment:
            for i in range(1, len(new_assignment[key])):
                new_assignment[key][i] = ind_assignment[key][i] \
                                         + f * (pop_best_assignment[key][i] - ind_assignment[key][i]) \
                                         + f * (ind1_assignment[key][i] - ind2_assignment[key][i]) \
                                         + f * (archive_best_assignment[key][i] - ind_assignment[key][i])

        # mutation of route structure --> reverse the satellite order in assignment chromosome
        satellite_order = [new_assignment[key][0] for key in new_assignment]
        for key in new_assignment:
            new_assignment[key][0] = satellite_order[-1]
            satellite_order.pop()

        depot_satellite_route = self.depot_satellite_route(new_assignment)
        satellite_customer_route = self.satellite_customer_route(new_assignment)
        new_ind = [new_assignment, depot_satellite_route, satellite_customer_route]
        new_ind += [[self.obj_time(new_ind), self.obj_satisfaction_equity(new_ind)[0],
                    self.obj_satisfaction_equity(new_ind)[1]]]
        return(new_ind)

    def not_feasible(self, ind):
        assignment, depot_satellite_route, satellite_customer_route = ind[0], ind[1], ind[2]

        # depot violation value: production amount exceed depot supply.
        d_value = 0
        for production_id in self.depot:
            production_amount = sum(assignment[key][production_id + 1] for key in assignment)
            production_amount_minus_supply = production_amount - self.depot[production_id][1]
            d_value += production_amount_minus_supply if production_amount_minus_supply > 0 else 0

        # satellite violation value: production amount exceed satellite capacity.
        s_value = 0
        satellite_production_amount = self.satellite_production_amount(assignment)
        for stl in satellite_production_amount:
            production_amount_minus_cap = sum(satellite_production_amount[stl]) - self.satellite[stl][1]
            s_value += production_amount_minus_cap if production_amount_minus_cap > 0 else 0

        # customer violation value: production amount exceed customer need or production amount is negative.
        c_value = 0
        for customer in assignment:
            for production_id in self.depot:
                if assignment[customer][production_id + 1] > 0:
                    exceed_customer_demand = assignment[customer][production_id + 1] - self.customer[customer][1][production_id]
                    c_value += exceed_customer_demand if exceed_customer_demand > 0 else 0
                else:
                    c_value += abs(assignment[customer][production_id + 1])

        # vehicle number violation value
        v_value = 0
        used_vehicle1_minus_num = sum([len(depot_satellite_route[key]) for key in depot_satellite_route]) - self.vehicle1_num
        v_value += used_vehicle1_minus_num if used_vehicle1_minus_num > 0 else 0
        used_vehicle2_minus_num = sum([len(satellite_customer_route[key]) for key in satellite_customer_route]) - self.vehicle1_num
        v_value += used_vehicle2_minus_num if used_vehicle2_minus_num > 0 else 0

        # weighted violation value
        # TODO parameter ？
        violation_value = self.not_feasible_weigh['depot'] * d_value \
                        + self.not_feasible_weigh['satellite'] * s_value \
                        + self.not_feasible_weigh['customer'] * c_value \
                        + self.not_feasible_weigh['vehicle'] * v_value
        return(violation_value)

    def constraint_choose(self, obj_index, ind0, ind1):
        if not self.not_feasible(ind0) and not self.not_feasible(ind1):
            if ind0[3][obj_index] < ind1[3][obj_index]:
                chosen_one = ind0[:]
            else:
                chosen_one = ind1[:]
        elif self.not_feasible(ind0) and not self.not_feasible(ind1):
            chosen_one = ind1[:]
        elif not self.not_feasible(ind0) and self.not_feasible(ind1):
            chosen_one = ind0[:]
        else:
            # TODO parameter ?
            if ind0[3][obj_index] + self.violation_weigh * self.not_feasible(ind0) \
             < ind1[3][obj_index] + self.violation_weigh * self.not_feasible(ind1):
                chosen_one = ind0[:]
            else:
                chosen_one = ind1[:]
        return(chosen_one)

    def single_objective_selection(self, obj_index, ind1, ind2, pop):
        if random.random() < self.mutt_prob:
            temp_ind1, temp_ind2 = self.mutation(ind1, pop), self.mutation(ind2, pop)
        else:
            temp_ind1, temp_ind2 = ind1[:], ind2[:]
        temp_pair = self.crossover(temp_ind1, temp_ind2)
        offspring = []
        offspring.append(self.constraint_choose(obj_index, temp_pair[0], ind1))
        offspring.append(self.constraint_choose(obj_index, temp_pair[1], ind2))
        return(offspring)

    def single_objective_evolution(self, obj_index, pop):
        # mu + lambda evolution strategy
        # input: population & evaluate function
        # output: the best feasible individual & offspring population
        offspring_population = []
        pairs = [random.sample(pop, 2) for _ in range(int(self.offspring_size / 2))]
        print(pairs)
        for pair in pairs[0:int(self.offspring_size/2)]:
            for a in self.single_objective_selection(obj_index, pair[0], pair[1], pop):
                offspring_population.append(a)
        temp_ind_li = pop + offspring_population
        temp_ind_li.sort(key=lambda ind: ind[3][obj_index])
        # return the best k individual
        return(temp_ind_li[:self.k])

    def a_dominate_b(self, ind_a, ind_b):
        for i in range(len(ind_a[3])):
            if ind_a[3][i] <= ind_b[3][i]:
                continue
            else: return(False)
        if ind_a[3] != ind_b[3]:
            return (True)
        else:
            return (False)

    def non_dominated_set(self, pop):
        non_dominated_ind = []
        dominated_ind = []
        rank = [1] * len(pop)
        for i in range(len(pop)):
            temp_count = 0
            for j in range(len(pop)):
                if self.a_dominate_b(pop[j], pop[i]):
                    temp_count += 1
            if temp_count == 0:
                non_dominated_ind.append(pop[i])
            else:
                dominated_ind.append(pop[i])
        return(non_dominated_ind, dominated_ind)

    def education(self, ind):
        educated_ind = []
        return(educated_ind)

    def multi_objective_evolution(self, best_k_s):
        # TODO chose the best k ind in every species, and put them into the archive set.
        # The 'education method' is applied to the individuals in archive set.
        # The archive set is separated into subsets of dominated one and non-dominated one.
        temp_archive = []
        for li in best_k_s:
            temp_archive += li
        res = self.non_dominated_set(temp_archive)
        non_dominated_ind, dominated_ind = res[0], res[1]
        if len(non_dominated_ind) <= self.archive_size:
            return(non_dominated_ind)
        else:
            return(non_dominated_ind[:self.archive_size])


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
    ini_pop = v.rand_pop()
    for i in range(30):
        best_k_s = []
        for obj_index in range(3):
            best_k_s.append(v.single_objective_evolution(0, ini_pop))
        archive = v.multi_objective_evolution(best_k_s)
    print(archive)




if __name__ == '__main__':
    main()
    # profile.run('main()')

# TODO add the "infeasible management and parameter setting" section in paper.
