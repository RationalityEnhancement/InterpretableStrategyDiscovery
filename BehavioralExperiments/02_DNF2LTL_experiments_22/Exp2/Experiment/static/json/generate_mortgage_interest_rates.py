import json
import numpy as np
import random
# n = 5
# for n_trials in range(0, n):

def generate_values(mu, sigma, decimals):
    var = np.random.normal(mu, sigma, 1)[0]
    if var < 0:
        var = 0.0
    var = np.round(var, decimals)
    return var


json_data = []

for _ in range(50):
    temp_dict = {"increasing": [generate_values(0.5, 0.2, 2), generate_values(1.5, 0.2, 2), generate_values(2.5, 0.2, 2)],
                 "decreasing": [generate_values(2.5, 0.2, 2), generate_values(1.5, 0.2, 2), generate_values(0.5, 0.2, 2)],
                 "constant": [generate_values(1, 0.1, 2), generate_values(1, 0.1, 2), generate_values(1, 0.1, 2)]}

    # shuffling values
    temp = list(temp_dict.items())
    random.shuffle(temp)

    # convert back to dict to get the index of the best plan
    dict_back = dict(temp)
    best_plan_index = list(dict_back.keys()).index("decreasing")
    if best_plan_index == 0:
        best_plan = "Plan A"
    elif best_plan_index == 1:
        best_plan = "Plan B"
    elif best_plan_index == 2:
        best_plan = "Plan C"
    else:
        print("Plan undefined")

    # get the interest rates and create a list
    value_list = []
    for key, value in temp:
        # print(key, ":", value)
        value_list.append(value)

    interest_rates_values = [item for sublist in value_list for item in sublist]

    # turn interest rate values into str
    interest_rates_values_str = [str(i) for i in interest_rates_values]

    # add % to all items in list
    interest_rates_values_str = [s + '%' for s in interest_rates_values_str]

    # append data into the json_list
    json_data.append({"interest_rate_values": interest_rates_values_str,
                  "best_plan": best_plan})

with open("mortgage.json", "w", encoding="utf-8") as f:
    json.dump(json_data, f)
