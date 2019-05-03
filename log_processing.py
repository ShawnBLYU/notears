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
time1_accept_sum_list = []
time2_running_sum_list = []
time2_accept_sum_list = []
w_change_list = []
num_accept = 0
num_reject = 0
    
for i in results_list:
    if i[0] == "time1":
        time1_running_sum += i[1]
        time1_sum += i[1]
    elif i[0] == "time2":
        time2_running_sum += i[1]
        time2_sum += i[1]
    elif i[0] == "accepted" or i[0] == "rejected":

        if i[0] == "accepted":
            time1_accept_sum_list.append(time1_running_sum)
            time2_accept_sum_list.append(time2_running_sum)
            num_accept += 1
        elif i[0] == "rejected":
            num_reject += 1

        time1_running_sum_list.append(time1_running_sum)
        time1_running_sum = 0
        time2_running_sum_list.append(time2_running_sum)
        time2_running_sum = 0
    else:
        w_change_list.append(i[1])


plt.plot(np.arange(len(time1_running_sum_list)), time1_running_sum_list, label='Time1')
plt.plot(np.arange(len(time2_running_sum_list)), time2_running_sum_list, label='Time2')
plt.legend()
plt.xlabel("Number of Updates")
plt.ylabel("Time (in seconds)")
plt.title("Time1 vs Time2")
plt.savefig("time1_vs_time2.png")

plt.clf()
plt.cla()
plt.plot(np.arange(len(w_change_list)), w_change_list)
plt.xlabel("Number of Accepted Updates")
plt.ylabel("L2 norm of change")
plt.title("Changes in W")
plt.savefig("w_change.png")

plt.clf()
plt.cla()
plt.plot(np.arange(len(time1_accept_sum_list)), time1_accept_sum_list, label='Time1')
plt.plot(np.arange(len(time2_accept_sum_list)), time2_accept_sum_list, label='Time2')
plt.legend()
plt.xlabel("Number of Accepted Updates")
plt.ylabel("Time (in seconds)")
plt.title("Time1 vs Time2")
plt.savefig("time1_vs_time2_accept.png")

print("There are {} accepts and {} rejects".format(num_accept, num_reject))
print("Proportion of accept is {}".format(num_accept/(num_accept + num_reject)))
print("Total time1 = {}, total time2 = {}".format(time1_sum, time2_sum))
print("Proportion of time2 is {}".format(time2_sum/(time1_sum + time2_sum)))
