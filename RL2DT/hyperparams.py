from MCRL.planning_strategies import strategy_dict
from MCRL.pure_planning_strategies import strategy_dict as strategy_dict_pure

## parameters defining the Mouselab environment
DISTRIBUTION = {1: [-8,-4,4,8],   
                2: [-8,-4,4,8],   
                3: [-8,-4,4,8],   
                4: [-48, -24, 24, 48], 
                5: [-48, -24, 24, 48]}
BRANCHING = [3,1,2]
ENV_TYPE = 'constant_paper'
TERM = 0
MAX_VAL = 48
NUM_ACTIONS = 13

## hyperparameters for generating Mouselab demonstrations
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

## hyperparameter for loading Mouselab demonstrations
BASE_NUM_DEMOS = 64

## (hyper)parameters for AI-Interpret
NUM_PROGRAMS = 14206
NUM_ROLLOUTS = 100000
NUM_CLUSTERS = list(range(1,26)) + [30,40,50,75,100,200,300]
ORDER = 'value'
SPLIT = 0.7
CLUSTER_DEPTH = 6
MAX_DEPTH = 5
ASPIRATION_LEVEL = 0.7
TOLERANCE = 0.025
REJECT_VAL = 0#0.025 #set to 0 for interpret_binary!
