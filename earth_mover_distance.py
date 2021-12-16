from random import choice
import math
import numpy as np
import os 
from collections import Counter
from collections import defaultdict
from ortools.linear_solver import pywraplp
import logging
logging.basicConfig(filename='earth_mover_distance_movie5.log', filemode='w', format='%(message)s', level=logging.DEBUG)


def list_of_tuples(points):
    return [tuple(point) for point in points]

'''
A python implementation of the Earthmover distance metric.
'''
def euclidean_distance(x, y):
    return math.sqrt(sum((a - b)**2 for (a, b) in zip(x, y)))


def earthmover_distance(p1, p2):
    '''
    Output the Earthmover distance between the two given points.
    Arguments:
     - p1: an iterable of hashable iterables of numbers (i.e., list of tuples)
     - p2: an iterable of hashable iterables of numbers (i.e., list of tuples)
    '''

    dist1 = {x: float(count) / len(p1) for (x, count) in Counter(p1).items()}
    dist2 = {x: float(count) / len(p2) for (x, count) in Counter(p2).items()}
    solver = pywraplp.Solver('earthmover_distance', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

    variables = dict()

    # for each pile in dist1, the constraint that says all the dirt must leave this pile
    dirt_leaving_constraints = defaultdict(lambda: 0)

    # for each hole in dist2, the constraint that says this hole must be filled
    dirt_filling_constraints = defaultdict(lambda: 0)

    # the objective
    objective = solver.Objective()
    objective.SetMinimization()

    for (x, dirt_at_x) in dist1.items():
        for (y, capacity_of_y) in dist2.items():
            amount_to_move_x_y = solver.NumVar(0, solver.infinity(), 'z_{%s, %s}' % (x, y))
            variables[(x, y)] = amount_to_move_x_y
            dirt_leaving_constraints[x] += amount_to_move_x_y
            dirt_filling_constraints[y] += amount_to_move_x_y
            objective.SetCoefficient(amount_to_move_x_y, euclidean_distance(x, y))

    for x, linear_combination in dirt_leaving_constraints.items():
        solver.Add(linear_combination == dist1[x])

    for y, linear_combination in dirt_filling_constraints.items():
        solver.Add(linear_combination == dist2[y])

    status = solver.Solve()
    if status not in [solver.OPTIMAL, solver.FEASIBLE]:
        raise Exception('Unable to find feasible solution')

    # for ((x, y), variable) in variables.items():
    #     if variable.solution_value() != 0:
    #         cost = euclidean_distance(x, y) * variable.solution_value()
#             print("move {} dirt from {} to {} for a cost of {}".format(
#                 variable.solution_value(), x, y, cost))

    return objective.Value()

# models = [
#     "raw_unnormalized.npy",
    # "data_forrest_paper_10pt_hiddendim64_bs64_mlp_md_mse_reg_lam1.0_manilam0.1_phate_shf_2_model_on_test_data_e4000_amlp_PHATE.npy",
    # "data_forrest_paper_10pt_hiddendim64_bs64_mlp_md_mse_reg_lam5.0_manilam0.1_phate_shf_model_on_test_data_e4000_amlp_PHATE.npy",
    # "data_forrest_paper_10pt_hiddendim64_bs64_mlp_md_mse_reg_lam10.0_manilam0.1_phate_shf_model_on_test_data_e4000_amlp_PHATE.npy",
    # "data_forrest_paper_10pt_hiddendim64_bs64_mlp_md_mse_reg_lam100.0_manilam0.1_phate_shf_2_model_on_test_data_e4000_amlp_PHATE.npy",
    
# ] 
# test_range = 624

models = [
    # "movie_raw_normalized.npy",
    "moviedata_forrest_paper_10pt_hiddendim64_bs64_mlp_md_mse_reg_lam0.1_manilam0.1_phate_shf_2half_7_model_on_2half_data_e4000_amlp_PHATE.npy",
    "moviedata_forrest_paper_10pt_hiddendim64_bs64_mlp_md_mse_reg_lam1.0_manilam0.1_phate_shf_2half_1_model_on_2half_data_e4000_amlp_PHATE.npy",
    "moviedata_forrest_paper_10pt_hiddendim64_bs64_mlp_md_mse_reg_lam5.0_manilam0.1_phate_shf_2half_1_model_on_2half_data_e4000_amlp_PHATE.npy"
]
test_range = 1799


patient_ids = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15]

# for model in models:
#     data_phate = np.load(os.path.join('phate', model))
#     point_sets = data_phate.reshape(len(patient_ids), test_range, -1)
    
#     logging.info(point_sets.shape)

#     used = []
#     distances = []
#     for i in range(10):
#         j = choice([x for x in range(10) if x != i and x not in used])
#         logging.info(f'subject {i} -- subject {j}')
#         d = earthmover_distance(list_of_tuples(point_sets[i]), list_of_tuples(point_sets[j]))
#         distances.append(d)
#         used.append(j)

#     logging.info(np.array([distances]).mean())


used = []
distances = {}
for model in models:
    distances[model] = []

model_phates = []
for model in models:
    data_phate = np.load(os.path.join('phate', model))
    point_sets = data_phate.reshape(len(patient_ids), test_range, -1)
    model_phates.append(point_sets)

for i in range(10):
    candidates = [x for x in range(10) if x != i and x not in used]
    if candidates:
        j = choice(candidates)
    else:
        j = choice([x for x in range(10)])
    logging.info(f'subject {i} -- subject {j}')

    for model, point_sets in zip(models, model_phates):
        p1 = list_of_tuples(point_sets[i])
        p2 = list_of_tuples(point_sets[j])
        d = earthmover_distance(p1, p2)
        distances[model].append(d)
    
    used.append(j)


for model in models:
    logging.info(model)
    logging.info(np.array(distances[model]).mean())