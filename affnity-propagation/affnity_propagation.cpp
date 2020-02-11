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


pair<double, double> zero_pair(0., 0.);
pair<double, double> default_maxes(-(double) INFINITY, -(double) INFINITY);


void update_matrix_r(SparseMatrix<double>& a, SparseMatrix<double>& s, SparseMatrix<double>& r, unordered_map<int, pair<double, double>>& double_maxes) {
    cout << "1 ";
    cout.flush();

    // pre-compute maxes for matrix R
    for (auto& row: s.data) {
        for (auto& pair: row.second) {
            int j = row.first;
            int k = pair.first;

            auto& maxes = double_maxes[j];
            auto edge_value = a.get(j, k) + pair.second;

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
    }

    cout << "2 ";
    cout.flush();

    // update matrix R
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
}


void update_matrix_a(SparseMatrix<double>& a, SparseMatrix<double>& s, SparseMatrix<double>& r, unordered_map<int, double>& sums) {
    cout << "3 ";
    cout.flush();

    // pre-compute sums for matrix A
    for (auto& row: s.data) {
        for (auto& pair: row.second) {
            int j = row.first;
            int k = pair.first;

            sums[k] += max(0., r.get(j, k));
        }
    }

    cout << "4 ";
    cout.flush();

    // update matrix A
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
}


void update_matrix_a_diag(SparseMatrix<double>& a, SparseMatrix<double>& s, SparseMatrix<double>& r, unordered_map<int, double>& sums, int size) {
    cout << "5 ";
    cout.flush();

    // pre-compute sums for diag of matrix A
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
    cout.flush();

    // update diag of matrix A
    for (int k = 0; k < size; ++k) {
        a.set(k, k, sums[k]);
    }
}


void iteration(SparseMatrix<double>& a, SparseMatrix<double>& s, SparseMatrix<double>& r, int size) {
    unordered_map<int, pair<double, double>> double_maxes;

    update_matrix_r(a, s, r, double_maxes);

    unordered_map<int, double> sums;

    update_matrix_a(a, s, r, sums);

    sums.clear();

    update_matrix_a_diag(a, s, r, sums, size);
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


int read_dataset(const char* dataset_filename, SparseMatrix<double>& s) {
    int size = -1;

    fstream dataset_file(dataset_filename, std::ios_base::in);
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
    dataset_file.close();
    return size;
}


void set_self_similarity(SparseMatrix<double>& s, int size, double value) {
    for (int i = 0; i < size; ++i) {
        s.set(i, i, value);
    }
}


void write_result_clusters(const char* out_filename, vector<pair<int, double>>& result) {
    fstream out_file(out_filename, ios_base::out);
    for (auto& pair: result) {
        out_file << pair.first << " ";
    }
    out_file << endl;
    out_file.close();
}


int main () {
    SparseMatrix<double> a, s, r;

    auto start = time(nullptr);

    int size = read_dataset("edges.txt", s);
    set_self_similarity(s, size, -2.);

    cout << "Start AP" << endl;

    for (int i = 0; i < 1; ++i) {
        cout << "Iteration " << i << endl;
        iteration(a, s, r, size);
    }
    auto result = get_result(a, s, r, size);

    write_result_clusters("clusters.txt", result);

    return 0;
}
