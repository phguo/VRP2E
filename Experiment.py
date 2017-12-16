from LRP2E import main as loop
from DataPreprocessing import load_instance_json as instances

PARAMETERS = {'pop_size': 500, 'offspring_size': 300,
              'archive_size': 400, 'k': 300,
              'obj_num': 3, 'f': 0.05,
              'mutt_prob': 0.05, 'cross_prob': 0.5,
              'violation_weigh': 0.5, 'not_feasible_weigh': {'depot':0.2, 'satellite':0.2, 'customer':0.2, 'vehicle':0.4},
              'iter_times': 50}

for ins in instances():
    loop(ins, PARAMETERS)
    # TODO A Bug when run rand_ind
