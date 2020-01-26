#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>

using namespace std;

namespace std {
    template<>
    struct hash<std::pair<int, int>> {
        inline size_t operator()(const std::pair<int, int> &v) const {
            std::hash<int> int_hasher;
            return int_hasher(v.first) ^ int_hasher(v.second);
        }
    };
}

using dict = unordered_map<pair<int, int>, double>;


struct Triple {
    double first, second, third;

    bool operator==(const Triple& other) const {
        return other.first == first && other.second == second && other.third == third;
    }
};


void print_dict(dict& s, int sz) {
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            if (s[make_pair(i, j)] > 0) {
                cout << " " << s[make_pair(i, j)] << " ";
            } else if (s[make_pair(i, j)] < 0) {
                cout << s[make_pair(i, j)] << " ";
            } else {
                cout << " 0 ";
            }
        }
        cout << endl;
    }
}


void iteration(dict& a, dict& s, dict& r, int size) {
    unordered_map<int, pair<double, double>> double_maxes;
    pair<double, double > zero_pair(0., 0.);
    pair<double, double > default_maxes(-(double)INFINITY, -(double)INFINITY);

//    cout << 1 << endl;
    // pre-compute maxes for (1)
    for (auto& edge_pair: s) {
        auto& maxes = double_maxes[edge_pair.first.first];
        auto edge_value = a[edge_pair.first] + edge_pair.second;

        if (maxes == zero_pair) {
            maxes = default_maxes;
        }
        if (edge_value >= maxes.first) {
            maxes.second = maxes.first;
            maxes.first = edge_value;
        } else if (edge_value > maxes.second) {
            maxes.second = edge_value;
        }
    }

//    cout << 2 << endl;
    // update matrix R (1)
    for (auto& edge_pair: s) {
        auto& maxes = double_maxes[edge_pair.first.second];
        auto edge_value = a[edge_pair.first] + edge_pair.second;

        double max_value;

        if (edge_value < maxes.first) {
            max_value = maxes.first;
        } else {
            max_value = maxes.second;
        }
        r[edge_pair.first] = edge_pair.second - max_value;
    }

    unordered_map<int, double> sums;

//    cout << 3 << endl;
    // pre-compute sums for (2)
    for (auto& edge_pair: s) {
        sums[edge_pair.first.second] += max(0., r[edge_pair.first]);
    }

//    cout << 4 << endl;
    // update matrix A (2)
    int count = 0;
    for (auto& edge_pair: s) {
        if (edge_pair.first.first == edge_pair.first.second) {
            continue;
        }
//        cout << count++ / (float)s.size() << endl;
        a[edge_pair.first] = min(0., r[make_pair(edge_pair.first.second, edge_pair.first.second)] + sums[edge_pair.first.second] - r[edge_pair.first]);
//        cout << "new val: " << r[make_pair(edge_pair.first.second, edge_pair.first.second)] << " " << a[edge_pair.first] << endl;
    }

    sums.clear();

//    cout << 5 << endl;
    // pre-compute sums for (3)
    for (auto& edge_pair: s) {
        if (edge_pair.first.first == edge_pair.first.second) {
            continue;
        }
        sums[edge_pair.first.second] += max(0., r[edge_pair.first]);
    }

//    cout << 6 << endl;
    for (int k = 0; k < size; ++k) {
        a[make_pair(k, k)] = sums[k];
    }
}


vector<pair<int, double>> get_result(dict& a, dict& s, dict& r, int size) {
    vector<pair<int, double>> c(size, make_pair(-1, -(double)INFINITY));

    for (auto& edge_pair: s) {
        int i = edge_pair.first.first;
        int k = edge_pair.first.second;

        double value = a[edge_pair.first] + r[edge_pair.first];

        if (value > c[i].second) {
            c[i].second = value;
            c[i].first = k;
        }
    }

    return c;
}


void print_result(vector<pair<int, double>>& result) {
    for (auto& pair: result) {
        cout << pair.first << " ";
    }
    cout << endl;
}


int main () {
//    srand(time(nullptr));
    dict a, s, r;

    const int size = 10;
    int edges_count = 10;
    auto start = time(nullptr);

    for (int i = 0; i < edges_count; i++) {
        int x = rand() % size;
        int y = rand() % size;
        s[make_pair(x, y)] = 1.;
        s[make_pair(y, x)] = 1.;
        s[make_pair(0, x)] = 1.;
        s[make_pair(x, 0)] = 1.;
    }

    for (int i = 0; i < size; ++i) {
        s[make_pair(i, i)] = -2.;
    }

    cout << "Start AP" << endl;

    for (int i = 0; i < 10; ++i) {
        iteration(a, s, r, size);
    }
    auto result = get_result(a, s, r, size);
    print_result(result);

//    cout << endl;
    cout << time(nullptr) - start << endl;
    return 0;
}
