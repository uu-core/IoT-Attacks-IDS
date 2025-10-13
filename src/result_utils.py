
from itertools import accumulate


def compute_BWT(performance_stability, train_domain_order):
    bwt_dict = {}
    for k, vals in performance_stability.items():
        if len(vals) < 2:
            continue
        f = vals[0]
        bwt_dict[k] = [v-f for v in vals[1:]]
    
    bwt_values, bwt_values_dict = avg_bwt_per_domain(bwt_dict, train_domain_order)
    return bwt_values, bwt_dict, bwt_values_dict
def avg_bwt_per_domain(data, domain_order):
    if not data:
        return []
    
    max_len = max((len(v) for v in data.values()), default=0)
    out = []
    # offset = how far from the end (1 = last element)
    for offset in range(max_len, 0, -1):
        col = [vals[-offset] for vals in data.values() if len(vals) >= offset]
        out.append(sum(col) / len(col))

    keys = domain_order[1: 1+ len(out)]
    bwt_values_dict = {k: v for k, v in zip(keys, out)}
    
    return out, bwt_values_dict


def compute_plasticity(performance_plasticity, domain_order):
    plasticity_dict = {}
    for k, vals in performance_plasticity.items():
        if len(vals) == 1:
            continue
 
        plasticity_dict[k] = vals[1] - vals[0]
    plasticity_values = avg_plasticity_per_domain(plasticity_dict, domain_order)    


    return plasticity_values, plasticity_dict

def avg_plasticity_per_domain(data: dict, domain_order: list) -> list[float]:
    vals = []

    for k in domain_order:
        if k in data:
            vals.append(data[k])

    cum_sum = list(accumulate(vals))

    return [s/(i+1) for i, s in enumerate(cum_sum)]


def calculate_cost():
    cost = 0.0
    # Implement cost calculation logic here
    return cost


