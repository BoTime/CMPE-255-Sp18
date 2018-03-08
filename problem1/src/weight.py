weight = [0.3577, 0.1813, 0.2340, 0.3521, 0.5033]
total_weight = 0.0

for w in weight:
    total_weight += w

for i in range(5):
    weight[i] /= total_weight

print weight