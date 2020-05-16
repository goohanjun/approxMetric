import pickle
import numpy as np


if __name__ == "__main__":
    print("Analyze the result")
    size = 500
    results = {}
    with open(f"./wmd_results_{size}.pkl", 'rb') as f:
        results = pickle.load(f)

    dist_types = []

    for k, _ in results[(0, 1)].items():
        if 'dist_' in k:
            dist_types.append(k.replace('dist_', ''))
    print(dist_types)

    times = np.zeros((len(dist_types), size, size))
    for (i, j), v in results.items():
        for dist_idx, dist_type in enumerate(dist_types):
            times[dist_idx, j, i] = times[dist_idx, i, j] = v[f'time_{dist_type}']

    for dist_idx, dist_type in enumerate(dist_types):
        print(dist_type, np.mean(times[dist_idx, :, :]))

