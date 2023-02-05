import pickle
import matplotlib.pyplot as plt
import numpy as np

# load the pickle file
data = pickle.load(open('/robodata/haresh92/spot-vrl/spot_data/RCA_test.pkl', 'rb'))

costs, tlabels = data['costs'], data['tlabels']

# normalize the cost between 0 and 1
costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))

# find mean costs for each label
cost_dict = {}
for tcost, tlabel in zip(costs, tlabels):
    if tlabel not in cost_dict:
        cost_dict[tlabel] = []
    cost_dict[tlabel].append(tcost)
    
for key in cost_dict.keys():
    cost_dict[key] = np.asarray(cost_dict[key])
    cost_dict[key] = np.mean(cost_dict[key])


# softmax normalization
tsum = np.sum([np.exp(cost_dict[key]/0.01) for key in cost_dict.keys()])
cost_dict = {key: np.exp(cost_dict[key]/0.01)/tsum for key in cost_dict.keys()}
print(cost_dict)

    
# multiply individual costs by the mean cost for that label
for i in range(len(costs)):
    costs[i] = costs[i] * cost_dict[tlabels[i]]

# plot the labels in x axis and the mean cost with std in y axis
mean_cost = np.mean(costs)
std_cost = np.std(costs)
plt.bar(range(len(cost_dict)), list(cost_dict.values()), yerr=std_cost, align='center', alpha=0.5, ecolor='black', capsize=10)
plt.xticks(range(len(cost_dict)), list(cost_dict.keys()), rotation=45)
plt.ylabel('Cost')
plt.title('Costs for different labels')
plt.tight_layout()
plt.savefig('rca_costs.png')

    
