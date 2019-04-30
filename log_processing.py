import glog as log
import matplotlib.pyplot as plt
import sys
import numpy as np

log_address = sys.argv[1]
keywords = ["time1", "time2", "accepted", "rejected", "change"]
def process_numbers(l):
    return np.float(l.split(" = ")[-1])

results_list = []
with open(log_address, "r") as f:
    num_accepts = 0
    num_rejects = 0
    for l in f:
        for w in keywords:
            if l.find(w) > 0:
                if w in ["time1", "time2", "change"]:
                    results_list.append((w, process_numbers(l)))
                elif w == "accepted":
                    results_list.append((w, num_accepts))
                    num_accepts += 1
                elif w == "rejected":
                    results_list.append((w, num_rejects))
                    num_rejects += 1

time1_running_sum = 0
time2_running_sum = 0
time1_sum = 0
time2_sum = 0
time1_running_sum_list = []
time2_running_sum_list = []
w_change_list = []
    
for i in results_list:


    if i[0] == "time1":
        time1_running_sum += i[1]
        time1_sum += i[1]
    elif i[0] == "time2":
        time2_running_sum += i[1]
        time2_sum += i[1]
    elif i[0] == "accepted" or i[0] == "rejected":
        time1_running_sum_list.append(time1_running_sum)
        time1_running_sum = 0
        time2_running_sum_list.append(time2_running_sum)
        time2_running_sum = 0
    else:
        w_change_list.append(i[1])

print(time1_running_sum_list)
print(time2_running_sum_list)
print(time1_sum)
print(time2_sum)
print(w_change_list)
