from collections import defaultdict, Counter

import numpy as np
from sklearn.model_selection import train_test_split


OUT_FILENAME = "clusters.txt" 
CHECKINS_FILENAME = "checkins.txt"
TOP_LOCATIONS_COUNT = 10
TEST_PERCENTAGE = 0.1


def recall(actual, predicted):
    return len(set(actual[:TOP_LOCATIONS_COUNT]).intersection(set(predicted[:TOP_LOCATIONS_COUNT])))


def read_checkins():
    checkins = []
    with open(CHECKINS_FILENAME) as file:
        for line in file:
            parts = line.split()
            checkins.append((int(parts[0]), int(parts[4])))
    return checkins


def read_clusters():
    with open(OUT_FILENAME) as file:
        return list(map(int, file.read().split()))


def extract_top(locations_stat: dict):
    return [
        x[0]
        for x in sorted(locations_stat.items(), key=lambda x: x[1])
        [-TOP_LOCATIONS_COUNT:]
    ]


def main():
    checkins = read_checkins()
    clusters = read_clusters()

    print("Total clusters: ", len(set(clusters)))
    clusters_stat = dict(Counter(clusters))
    print("Cluster sizes top 10:", sorted(clusters_stat.values(), reverse=True)[:10])
    hg_stat, header = np.histogram(np.array(list(clusters_stat.values())), bins=[1, 2, 3, 10, 100, 1000, 10000])
    print("Cluster count histogram:")
    print('\t'.join([str(x) for x in hg_stat]))
    print('\t'.join([str(x) for x in header[:-1]]))

    data = []
    for checkin in checkins:
        data.append([checkin[0], checkin[1], clusters[checkin[0]]])

    data = np.array(data)
    checked_in_users = np.unique(data[:, 0])

    train_users, test_users = train_test_split(checked_in_users, test_size=TEST_PERCENTAGE, shuffle=True)

    train_users_set = set(train_users)
    test_users_set = set(test_users)

    train_data = [row for row in data if row[0] in train_users_set]
    test_data = [row for row in data if row[0] in test_users_set]

    test_user_to_locations = defaultdict(list)

    for test_row in test_data:
        user_id, location_id, _ = test_row
        test_user_to_locations[user_id].append(location_id)

    locations_stat = defaultdict(int)

    for train_row in train_data:
        locations_stat[train_row[1]] += 1

    top_train_locations = extract_top(locations_stat)

    top_locations_in_clusters = defaultdict(lambda: defaultdict(int))

    for train_row in train_data:
        _, location_id, cluster_id = train_row
        top_locations_in_clusters[cluster_id][location_id] += 1

    cluster_score = 0
    train_total_score = 0

    cache = {}

    for test_user in test_users:
        cluster = clusters[test_user]
        user_locations = test_user_to_locations[test_user]

        if cluster in top_locations_in_clusters:
            if cluster not in cache:
                cache[cluster] = recall(extract_top(top_locations_in_clusters[cluster]), user_locations)

            cluster_score += cache[cluster]

        train_total_score += recall(top_train_locations, user_locations)

    k = len(test_users)

    cluster_score /= k
    train_total_score /= k

    print("Train total score: ", train_total_score)
    print("Cluster recommendation score: ", cluster_score)


if __name__ == '__main__':
    main()
