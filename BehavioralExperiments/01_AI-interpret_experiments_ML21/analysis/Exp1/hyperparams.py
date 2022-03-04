from MCRL.planning_strategies import strategy_dict
from MCRL.pure_planning_strategies import strategy_dict as strategy_dict_pure

MAX_DEMOS = 64
NUM_ACTIONS = 13
NUM_PROGRAMS = 20985
MAX_VAL = 48

DISTRIBUTION = {1: [-4, -2, 4, 2],
                2: [-8, -4, 4, 8],
                3: [-48, -24, 24, 48],
                4: [-48, -24, 24, 48], 
                5: [-48, -24, 24, 48]}
BRANCHING = [3,1,2]
ENV_TYPE = 'standard'

STRATEGIES = {"BRFS": strategy_dict[3],
              "DFS": strategy_dict[31],
              "Immediate": strategy_dict[23],
              "Final": strategy_dict[24],
              "BEFS": strategy_dict[25],
              "Optimal": strategy_dict[21],
              "NO": strategy_dict[17],
              "NO2": strategy_dict[19],
              "NO3": strategy_dict[18],
              "Inverse_NO2": strategy_dict[20],
              "NO4": strategy_dict[16],
              "copycat": strategy_dict[40]}
PURE_STRATEGIES = strategy_dict_pure
STRATEGY_NAME = "copycat"
STRATEGY = STRATEGIES[STRATEGY_NAME]
PURE_STRATEGY = PURE_STRATEGIES[STRATEGY_NAME]

NUM_CLUSTERS = 3
ORDER = 'biggest'
SPLIT = 0.7
CLUSTER_DEPTH = 5
MAX_DEPTH = 4
ASPIRATION_LEVEL = 1
