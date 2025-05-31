import numpy as np
import os

x = len(os.listdir(os.path.join('/storage2/datasets/jkarhade/CART_place_recognition','ocean_duck','color')))
print(x)

soft_positives_per_query = []

for i in range(x):
    min_x = max(0,i-5)
    max_x = min(x,i+5)
    cur_soft_positives = [i for i in range(min_x,max_x)]

    soft_positives_per_query.append(cur_soft_positives)

soft_positives_per_query = np.array(soft_positives_per_query)

save_path = (os.path.join('/storage2/datasets/jkarhade/CART_place_recognition','ocean_duck','soft_positives_per_query.npy'))
np.save(save_path,soft_positives_per_query)