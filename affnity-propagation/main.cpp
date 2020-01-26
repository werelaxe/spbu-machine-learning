#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>

using namespace std;


template <typename T>
struct SparseMatrix {
    T get(int i, int k) {
        if (data.find(i) != data.end()) {
            if (data[i].find(k) != data[i].end()) {
                return data[i][k];
            }
        }
        return T();
    }

    void set(int i, int k, T value) {
        data[i][k] = value;
    }

    unordered_map<int, unordered_map<int, T>> data;
};


void print_dict(SparseMatrix<double>& s, int sz) {
    for (int i = 0; i < sz; ++i) {
        for (int j = 0; j < sz; ++j) {
            double value = s.get(i, j);
            if (value > 0) {
                cout << " " << value << " ";
            } else if (value < 0) {
                cout << value << " ";
            } else {
                cout << " 0 ";
            }
        }
        cout << endl;
    }
    cout << endl;
}


void iteration(SparseMatrix<double>& a, SparseMatrix<double>& s, SparseMatrix<double>& r, int size) {
    unordered_map<int, pair<double, double>> double_maxes;
    pair<double, double> zero_pair(0., 0.);
    pair<double, double> default_maxes(-(double) INFINITY, -(double) INFINITY);

    cout << 1 << endl;

    for (auto& row: s.data) {
        for (auto& pair: row.second) {
            int j = row.first;
            int k = pair.first;

            auto& maxes = double_maxes[j];
            auto edge_value = a.get(j, k) + pair.second;

            if (maxes == zero_pair) {
                maxes = default_maxes;
            }
//            cout << "(" << i << ", " << k << ") ev: " << edge_value << ", fm: " << maxes.first << ", sm: " << maxes.second << endl;

            if (edge_value >= maxes.first) {
                maxes.second = maxes.first;
                maxes.first = edge_value;
            } else if (edge_value > maxes.second) {
                maxes.second = edge_value;
            }
        }
    }

//    for (int i = 0; i < 10; ++i) {
//        cout << i << " " << double_maxes[i].first << " " << double_maxes[i].second << endl;
//    }

    cout << 2 << endl;

//    exit(0);
//    int count = 0;
    // update matrix R (1)
    for (auto& row: s.data) {
        for (auto& pair: row.second) {
            int i = row.first;
            int k = pair.first;

            auto& maxes = double_maxes[i];
            auto edge_value = a.get(i, k) + pair.second;

            double max_value;

            if (edge_value < maxes.first) {
                max_value = maxes.first;
            } else {
                max_value = maxes.second;
            }
            r.set(i, k, pair.second - max_value);
        }
    }

    unordered_map<int, double> sums;

    cout << 3 << endl;
    // pre-compute sums for (2)
    for (auto& row: s.data) {
        for (auto& pair: row.second) {
            int j = row.first;
            int k = pair.first;

            sums[k] += max(0., r.get(j, k));
        }
    }

    cout << 4 << endl;
    // update matrix A (2)
    for (auto& row: s.data) {
        for (auto& pair: row.second) {
            int i = row.first;
            int k = pair.first;

            if (i == k) {
                continue;
            }

            a.set(i, k,min(0., r.get(k, k) + sums[k] - r.get(i, k) - r.get(k, k)));
        }
    }

    sums.clear();

    cout << 5 << endl;
    // pre-compute sums for (3)
    for (auto&  row: s.data) {
        for (auto& pair: row.second) {
            int j = row.first;
            int k = pair.first;

            if (j == k) {
                continue;
            }
            sums[k] += max(0., r.get(j, k));
        }
    }

    cout << 6 << endl;
    for (int k = 0; k < size; ++k) {
        a.set(k, k, sums[k]);
    }
}


vector<pair<int, double>> get_result(SparseMatrix<double>& a, SparseMatrix<double>& s, SparseMatrix<double>& r, int size) {
    vector<pair<int, double>> c(size, make_pair(-1, -(double)INFINITY));
    for (auto& row: s.data) {
        for (auto& pair: row.second) {
            int i = row.first;
            int k = pair.first;

            double value = a.get(i, k) + r.get(i, k);

            if (value > c[i].second) {
                c[i].second = value;
                c[i].first = k;
            }
        }
    }
    return c;
}


int main () {
    SparseMatrix<double> a, s, r;

    int size = -1;
    auto start = time(nullptr);

    fstream dataset_file("../edges.txt", std::ios_base::in);
    int x;
    bool flag = false;
    int buff = -1;

    int count = 0;

    while (dataset_file >> x) {
        size = max(size, x);
        if (flag) {
            s.set(buff, x, 1.);
            s.set(x, buff, 1.);
        } else {
            buff = x;
        }
        flag = !flag;
    }

    size++;
    cout << "Size: " << size << endl;

    for (int i = 0; i < size; ++i) {
        s.set(i, i, -1.5);
    }

    cout << "Start AP" << endl;

    for (int i = 0; i < 3; ++i) {
        iteration(a, s, r, size);
    }
    auto result = get_result(a, s, r, size);

    fstream out_file("../out.txt", ios_base::out);

    for (auto& pair: result) {
        out_file << pair.first << " ";
    }
    out_file << endl;

    out_file.close();
    dataset_file.close();
    return 0;
}
