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


class VRP2E:
    def __init__(self, instance, parameters):
        # data
        self.depot, self.satellite, self.customer = instance['depot'], instance['satellite'], instance['customer']
        self.vehicle1_cap, self.vehicle2_cap = instance['vehicle1_cap'], instance['vehicle2_cap']
        self.vehicle1_num, self.vehicle2_num = instance['vehicle1_num'], instance['vehicle2_num']
        self.satellite_cap = instance['satellite_cap']
        self.loc = {}
        for data in [self.depot, self.satellite, self.customer]:
            for i in data:
                self.loc[i] = data[i][0]

        # evolutionary algorithm parameters
        self.iter_times = parameters['iter_times']
        self.pop_size = parameters['pop_size']
        self.offspring_size = parameters['offspring_size']
        self.archive_size = parameters['archive_size']
        # self.obj_num = parameters['obj_num']
        self.k = parameters['k']  # self.pop_size
        self.f = parameters['f']
        self.mutt_prob = parameters['mutt_prob']
        # self.cross_prob = parameters['cross_prob']
        self.violation_weigh = parameters['violation_weigh']
        self.not_feasible_weigh = parameters['not_feasible_weigh']

    def satellite_production_amount(self, assignment):
        result_dic = {i: [0] * len(self.depot) for i in self.satellite}
        for i in self.satellite:
            for key in assignment:
                if i == assignment[key][0]:
                    for p in range(len(self.depot)):
                        result_dic[i][p] += assignment[key][p + 1]
        return (result_dic)

    def customer_production_total(self, assignment):
        result_dic = {i: np.sum(assignment[i][1:]) for i in self.customer}
        return (result_dic)

    def depot_satellite_route(self, assignment):
        # route depot -> satellite is generated according to the assignment.
        spa = self.satellite_production_amount(assignment)
        satellite_li = []
        for key in assignment:
            s = assignment[key][0]
            if s not in satellite_li:
                satellite_li.append(s)

        depot_satellite_route = {depot: [[]] for depot in self.depot}
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
        for key in depot_satellite_route:
            while [] in depot_satellite_route[key]:
                depot_satellite_route[key].remove([])
        return (depot_satellite_route)

    def satellite_customer_route(self, assignment):
        # route satellite -> customer is generated according to the assignment.
        cpt = self.customer_production_total(assignment)

        satellite_customer_assignment = {assignment[key][0]: [] for key in assignment}
        for stl in self.satellite:
            for key in assignment:
                if stl == assignment[key][0]:
                    satellite_customer_assignment[stl].append(key)

        satellite_customer_route = {satellite: [[]] for satellite in satellite_customer_assignment}
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
        return (satellite_customer_route)

    def rand_ind(self):
        # the total amount of every production is separated to some parts randomly.
        customer_li = [key for key in self.customer]
        random.shuffle(customer_li)

        assignment = OrderedDict({customer: [satellite]
                                  for customer, satellite
                                  in zip(customer_li,
                                         [random.choice([key for key in self.satellite]) for _ in
                                          range(len(self.customer))])})

        for key in assignment:
            for i in range(len(self.depot)):
                # FIXME
                mu = self.depot[i][1] / len(self.customer)
                sigma = mu / 3
                assignment[key].append(np.random.normal(mu, sigma, 1)[0])

        d_s = self.depot_satellite_route(assignment)
        s_c = self.satellite_customer_route(assignment)
        individual = [assignment, d_s, s_c]
        individual += [self.obj_value(individual)]
        individual += [self.not_feasible(individual)]
        return (individual)

    def rand_pop(self, obj_index):
        pop = [self.rand_ind() for _ in range(self.pop_size)]
        sorted_pop = sorted(pop, key=lambda ind: ind[3][obj_index])

        # standardize violation_value
        for ind in pop:
            ind += [self.standardize_not_feasible(ind, pop)]
        return (sorted_pop)

    def obj_value(self, ind):
        def obj_time(self, ind):
            obj_value = 0
            d_s, s_c = ind[1], ind[2]
            assert (set(d_s) & set(s_c) == set())
            temp_dic = copy.deepcopy(d_s)
            temp_dic.update(s_c)

            for key in temp_dic:
                for li in temp_dic[key]:
                    temp_li = [key] + li + [key]
                    for i in range(len(temp_li) - 1):
                        obj_value += math.sqrt(
                            (self.loc[temp_li[i]][0] - self.loc[temp_li[i + 1]][0]) ** 2
                            + (self.loc[temp_li[i]][1] - self.loc[temp_li[i + 1]][1]) ** 2)
            return (obj_value)

        def customer_satisfaction(self, ind):
            assign = ind[0]
            customer_satisfaction_dic = {i: [0] * len(self.depot) for i in self.customer}

            for key in assign:
                for j in range(len(self.depot)):
                    # TODO
                    s = assign[key][j + 1] / self.customer[key][1][j]
                    customer_satisfaction_dic[key][j] = s if s <= 1 else 1

            customer_satisfaction_dic = {i: [customer_satisfaction_dic[i], sum(customer_satisfaction_dic[i])]
                                         for i in customer_satisfaction_dic}  # different production share same weight
            return (customer_satisfaction_dic)

        def obj_satisfaction_equity(self, ind):
            customer_satisfaction_dic = customer_satisfaction(self, ind)
            temp_li = [customer_satisfaction_dic[i][1] for i in customer_satisfaction_dic]
            return (-np.sum(temp_li), np.var(temp_li))

        obj_t = obj_time(self, ind)
        obj_s_e = obj_satisfaction_equity(self, ind)
        return ([obj_t, obj_s_e[0], obj_s_e[1]])

    def crossover(self, ind0, ind1, pop):
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
        ind0_son += [self.obj_value(ind0_son)]
        ind1_son += [self.obj_value(ind1_son)]
        ind0_son += [self.not_feasible(ind0_son)]
        ind1_son += [self.not_feasible(ind1_son)]

        # standardize violation_value
        temp_pop = pop + [ind0_son] + [ind1_son]
        ind0_son += [self.standardize_not_feasible(ind0_son, temp_pop)]
        ind1_son += [self.standardize_not_feasible(ind1_son, temp_pop)]
        return (ind0_son, ind1_son)

    def mutation(self, ind, pop, archive):
        pop_best = random.choice(pop)
        for i in pop:
            if not i[5]:
                pop_best = i[:]
                break
        archive_best = random.choice(archive)  # just for test

        pop_best_assignment, archive_best_assignment = pop_best[0], archive_best[0]
        ind1_assignment, ind2_assignment = random.choice(pop)[0], random.choice(pop)[0]
        ind_assignment = ind[0]
        # new_assignment = OrderedDict({key: ind_assignment[key][:] for key in ind_assignment})
        new_assignment = copy.deepcopy(ind_assignment)

        # mutation of delivery amount --> follow the method of Wang(2016)(8)
        # TODO parameter ? coevolution trail vector
        for key in new_assignment:
            for i in range(1, len(new_assignment[key])):
                new_assignment[key][i] = ind_assignment[key][i] \
                                         + self.f * (pop_best_assignment[key][i] - ind_assignment[key][i]) \
                                         + self.f * (ind1_assignment[key][i] - ind2_assignment[key][i]) \
                                         + self.f * (archive_best_assignment[key][i] - ind_assignment[key][i])

        # mutation of route structure --> reverse the satellite order in assignment chromosome
        satellite_order = [new_assignment[key][0] for key in new_assignment]
        for key in new_assignment:
            new_assignment[key][0] = satellite_order[-1]
            satellite_order.pop()

        depot_satellite_route = self.depot_satellite_route(new_assignment)
        satellite_customer_route = self.satellite_customer_route(new_assignment)
        new_ind = [new_assignment, depot_satellite_route, satellite_customer_route]
        new_ind += [self.obj_value(new_ind)]
        new_ind += [self.not_feasible(new_ind)]

        # standardize violation_value
        temp_pop = pop + [new_ind]
        new_ind += [self.standardize_not_feasible(new_ind, temp_pop)]
        return (new_ind)

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
                    # TODO
                    # exceed_customer_demand = assignment[customer][production_id + 1] - self.customer[customer][1][
                    #     production_id]
                    # c_value += exceed_customer_demand if exceed_customer_demand > 0 else 0
                    c_value += 0
                else:
                    c_value += abs(assignment[customer][production_id + 1])

        # vehicle number violation value
        v_value = 0
        used_vehicle1_minus_num = sum(
            [len(depot_satellite_route[key]) for key in depot_satellite_route]) - self.vehicle1_num
        v_value += used_vehicle1_minus_num if used_vehicle1_minus_num > 0 else 0
        used_vehicle2_minus_num = sum(
            [len(satellite_customer_route[key]) for key in satellite_customer_route]) - self.vehicle1_num
        v_value += used_vehicle2_minus_num if used_vehicle2_minus_num > 0 else 0

        # weighted violation value
        # violation_value = self.not_feasible_weigh['depot'] * d_value \
        #                   + self.not_feasible_weigh['satellite'] * s_value \
        #                   + self.not_feasible_weigh['customer'] * c_value \
        #                   + self.not_feasible_weigh['vehicle'] * v_value

        # print(d_value, s_value, c_value, v_value)
        # return (violation_value)
        return([d_value, s_value, c_value, v_value])

    def standardize_not_feasible(self, ind, pop):
        # not_feasible should be a 1*4 list
        temp_pop = pop + [ind]
        d_value_li, s_value_li, c_value_li, v_value_li = [], [], [], []
        for i in temp_pop:
            not_feasible_li = i[4]
            d_value_li.append(not_feasible_li[0])
            s_value_li.append(not_feasible_li[1])
            c_value_li.append(not_feasible_li[2])
            v_value_li.append(not_feasible_li[3])

        violation_value = 0
        ind_not_feasible_li = ind[4]
        for i, li in enumerate([d_value_li, s_value_li, c_value_li, v_value_li]):
            if ind_not_feasible_li[i] == 0:
                pass
            elif min(li) == max(li):
                violation_value += 1
            elif not sum([abs(a) for a in li]) == 0:
                violation_value += (ind_not_feasible_li[i] - min(li)) / (max(li) - min(li))
        print(violation_value, ind_not_feasible_li)
        return(violation_value)

    def constraint_choose(self, obj_index, ind0, ind1):
        if not ind0[5] and not ind1[5]:
            # a feasible, b feasible
            if ind0[3][obj_index] < ind1[3][obj_index]:
                chosen_one = ind0[:]
            else:
                chosen_one = ind1[:]
        elif ind0[5] and not ind1[5]:
            # a not feasible, b feasible
            chosen_one = ind1[:]
        elif not ind0[5] and ind1[5]:
            # a feasible, b not feasible
            chosen_one = ind0[:]
        else:
            # a not feasible, b not feasible
            # if ind0[3][obj_index] + self.violation_weigh * ind0[5] \
            #         < ind1[3][obj_index] + self.violation_weigh * ind1[5]:
            if  ind0[5] < ind1[5]:
                chosen_one = ind0[:]
            else:
                chosen_one = ind1[:]
        return (chosen_one)

    def single_objective_selection(self, obj_index, ind1, ind2, pop, archive):
        # input 2 ind, output 2 ind
        if random.random() < self.mutt_prob:
            temp_ind1, temp_ind2 = self.mutation(ind1, pop, archive), self.mutation(ind2, pop, archive)
        else:
            temp_ind1, temp_ind2 = ind1[:], ind2[:]
        temp_pair = self.crossover(temp_ind1, temp_ind2, pop)
        offspring = []
        offspring.append(self.constraint_choose(obj_index, temp_pair[0], ind1))
        offspring.append(self.constraint_choose(obj_index, temp_pair[1], ind2))
        return (offspring)

    def single_objective_evolution(self, obj_index, pop, archive):
        # mu + lambda evolution strategy, the best one is preserved.
        # mu ~ pop_size, lambda ~ offspring_size
        # input: population & evaluate function
        # output: the best feasible individual & offspring population
        temp_pop = pop[:]
        offspring_population = []
        chose_ind = []
        for _ in range(int(self.offspring_size / 2)):
            ind = random.choice(temp_pop)
            chose_ind.append(ind[:])
            temp_pop.remove(ind)
        pairs = []
        while chose_ind != []:
            ind0 = random.choice(chose_ind)[:]
            chose_ind.remove(ind0)
            ind1 = random.choice(chose_ind)[:]
            chose_ind.remove(ind1)
            pairs.append((ind0, ind1))
        for pair in pairs:
            for a in self.single_objective_selection(obj_index, pair[0], pair[1], pop, archive):
                offspring_population.append(a)
        new_pop = temp_pop + offspring_population
        # remove duplicates
        # t_pop = []
        # for ind in new_pop:
        #     if ind not in temp_pop:
        #         temp_pop.append(ind)
        sorted_new_pop = sorted(new_pop, key=lambda ind: ind[3][obj_index])

        # preserve the best feasible ind
        sorted_new_pop.remove(sorted_new_pop[-1])
        for ind in sorted_new_pop:
            if not ind[5]:
                sorted_new_pop.append(ind)
                break
        sorted_new_pop = sorted(sorted_new_pop, key=lambda ind: ind[3][obj_index])

        # archive the k best feasible ind
        k_best = []
        for ind in sorted_new_pop:
            if not ind[5]:
                k_best.append(ind)
            if len(k_best) >= self.k:
                break
        return (k_best, sorted_new_pop)

    def a_dominate_b(self, ind_a, ind_b):
        for i in range(len(ind_a[3])):
            if ind_a[3][i] <= ind_b[3][i]:
                continue
            else:
                return (False)
        if ind_a[3] != ind_b[3]:
            return (True)
        else:
            return (False)

    def non_dominated_set(self, pop):
        non_dominated_ind = []
        dominated_ind = []
        for i in range(len(pop)):
            temp_count = 0
            for j in range(len(pop)):
                if self.a_dominate_b(pop[j], pop[i]):
                    temp_count += 1
            if temp_count == 0:
                non_dominated_ind.append(pop[i])
            else:
                dominated_ind.append(pop[i])
        return (non_dominated_ind, dominated_ind)

    def education(self, ind):
        # education method is applied to the k-best ind
        educated_ind = []
        return (educated_ind)

    def multi_objective_evolution(self, archive, best_k_s):
        # chose the best k ind in every species, and put them into the archive set.
        # The 'education method' is applied to the individuals in archive set.
        # The archive set is separated into subsets of dominated one and non-dominated one.
        temp_archive = archive + best_k_s
        # remove duplicates
        new_archive = []
        for ind in temp_archive:
            if not ind in new_archive:
                new_archive.append(ind)

        res = self.non_dominated_set(new_archive)
        non_dominated_ind, dominated_ind = res[0], res[1]
        if len(non_dominated_ind) <= self.archive_size:
            return (non_dominated_ind)
        else:
            # TODO select according to 'crowd distance' proposed by Deb
            return (non_dominated_ind[:self.archive_size])


def timer(func):
    def wrapTheFunction():
        start_time = time.clock()
        func()
        end_time = time.clock()
        print('time consuming:', end_time - start_time)

    return wrapTheFunction


# @timer
def main(instance, parameter):
    v = VRP2E(instance, parameter)
    non_dominated_archive = []
    best_k_s = []
    for i in range(3):
        ini_pop = v.rand_pop(i)
        obj_i_best_k, single_objective_offspring = v.single_objective_evolution(i, ini_pop, ini_pop)
        best_k_s += obj_i_best_k
    non_dominated_archive = v.multi_objective_evolution(non_dominated_archive, best_k_s)

    for _ in range(v.iter_times):
        # print(_)
        if non_dominated_archive == []:  # FIXME
            non_dominated_archive = single_objective_offspring[:]
        best_k_s = []
        for i in range(3):
            obj_i_best_k, single_objective_offspring = v.single_objective_evolution(i, single_objective_offspring,
                                                                                    non_dominated_archive)
            best_k_s += obj_i_best_k
        # temp_li = non_dominated_archive[:]
        non_dominated_archive = v.multi_objective_evolution(non_dominated_archive, best_k_s)
        # print(len([a for a in non_dominated_archive if a not in temp_li]), len(non_dominated_archive))
        #
        # for ind in non_dominated_archive:
        #     print(ind[-2:])
    return (non_dominated_archive)


# if __name__ == '__main__':
#     # randomly generated data
#     DEPOT_NUM, SATELLITE_NUM, CUSTOMER_NUM = 3, 5, 9
#     SATELLITE_CAP = float("inf")  # 500
#     VEHICLE1_NUM, VEHICLE2_NUM = 10, 8
#     VEHICLE1_CAP, VEHICLE2_CAP = 60, 60
#     DEPOT = {i: ((random.uniform(0, 10), random.uniform(0, 10)), 100)
#              for i in range(DEPOT_NUM)}
#     SATELLITE = {i: ((random.uniform(0, 10), random.uniform(0, 10)), 1000)
#                  for i in range(max(DEPOT) + 1, max(DEPOT) + 1 + SATELLITE_NUM)}
#     CUSTOMER = {i: ((random.uniform(0, 10), random.uniform(0, 10)), [20] * DEPOT_NUM)
#                 for i in range(max(SATELLITE) + 1, max(SATELLITE) + 1 + CUSTOMER_NUM)}
#     INSTANCE = {'depot': DEPOT, 'satellite': SATELLITE, 'customer': CUSTOMER,
#                 'vehicle1_num': VEHICLE1_NUM, 'vehicle2_num': VEHICLE2_NUM,
#                 'vehicle1_cap': VEHICLE1_CAP, 'vehicle2_cap': VEHICLE2_CAP,
#                 'satellite_cap': SATELLITE_CAP}
#     # parameters
#     PARAMETERS = {'pop_size': 500, 'offspring_size': 300,
#                   'archive_size': 400, 'k': 300,
#                   'obj_num': 3, 'f': 0.05,
#                   'mutt_prob': 0.05, 'cross_prob': 0.5,
#                   'violation_weigh': 0.5,
#                   'not_feasible_weigh': {'depot': 0.2, 'satellite': 0.2, 'customer': 0.2, 'vehicle': 0.4},
#                   'iter_times': 50}
#
#     main(INSTANCE, PARAMETERS)
#     # profile.run('main()')

# TODO add the "infeasible management and parameter setting" section in paper.
# TODO modify the depot-satellite route generate strategy (single depot VRP * len(self.depot)). paper
