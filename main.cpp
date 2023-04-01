#include <algorithm>
#include <ctime>
#include <iostream>
#include <map>
#include <tuple>
#include <vector>
#include <random>

using namespace std;
using vec = vector<int>;
using mat = vector<vec>;
using tensor = vector<mat>;

const int N = 2;
const int M = 2;
const int K = 2;

mt19937 rnd(clock());

tensor Zeros(size_t n, size_t m, size_t k) {
    return tensor(n, mat(m, vec(k)));
}

vec Rand(size_t n) {
    vec v(n);
    for (size_t i = 0; i < n; i++) {
        v[i] = (int)rnd() % 2;
    }
    return v;
}

vec Vectorize(const tensor &ts) {
    vec res;
    res.reserve(ts.size() * ts[0].size() * ts[0][0].size());
    for (const mat &mat1 : ts) {
        for (const vec &vec1 : mat1) {
            for (int elem : vec1) {
                res.push_back(elem);
            }
        }
    }
    return res;
}

vec operator-(const vec &v1, const vec &v2) {
    vec v(v1.size());
    for (size_t i = 0; i < v.size(); i++) {
        v[i] = v1[i] ^ v2[i];
    }
    return v;
}

int Sum(const vec &v) {
    int sum = 0;
    for (int elem : v) {
        sum += elem;
    }
    return sum;
}

tensor OuterProduct(const vec &v1, const vec &v2, const vec &v3) {
    auto res = Zeros(v1.size(), v2.size(), v3.size());
    for (size_t i = 0; i < v1.size(); i++) {
        for (size_t j = 0; j < v2.size(); j++) {
            for (size_t t = 0; t < v3.size(); t++) {
                res[i][j][t] = v1[i] * v2[j] * v3[t];
            }
        }
    }
    return res;
}

void PrintTensor(const tensor &ts) {
    for (const mat &mat1 : ts) {
        for (const vec &vec1 : mat1) {
            for (int elem : vec1) {
                cout << elem;
            }
            cout << ' ';
        }
        cout << endl;
    }
}

tensor GenerateTensor(size_t n, size_t k, size_t m) { // (n x k) * (k x m) -> (n x m)
    tensor res = Zeros(n * k, k * m, n * m);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < m; j++) {
            for (size_t t = 0; t < k; t++) {
                res[i * k + t][t * m + j][i * m + j] = 1;
            }
        }
    }
    return res;
}

int Weight(const vec &vectorized) {
    int sum = 0;
    for (int elem : vectorized) {
        sum += abs(elem);
    }
    return sum;
}

int main() {
    auto target = Vectorize(GenerateTensor(N, K, M));
    auto zeros = Vectorize(Zeros(N * K, K * M, N * M));
    const int mem_size = 1000;
    const int mult_factor = 1 << (N * K + K * M + N * M);
    vector<vec> layer = {target};
    int cnt_layers = Sum(target);
    for (int layer_id = 0; layer_id < cnt_layers; layer_id++) {
//        cout << "layer " << layer_id << endl;
//        for (const auto &v : layer) {
//            for (int elem : v) cout << elem;
//            cout << endl;
//        }
//        cout << endl;
        vector<vec> next_layer;
        for (const auto &v : layer) {
            for (size_t t = 0; t < mult_factor; t++) {
                vec v1(N * K), v2(K * M), v3(N * M);
                for (int bit = 0; bit < N * K + K * M + N * M; bit++) {
                    if (bit < N * K) v1[bit] = (t >> bit) & 1;
                    else if (bit < N * K + K * M) v2[bit - N * K] = (t >> bit) & 1;
                    else v3[bit - N * K - K * M] = (t >> bit) & 1;
                }
                auto prod = OuterProduct(v1, v2, v3);
                next_layer.push_back(v - Vectorize(prod));
            }
        }
        sort(next_layer.begin(), next_layer.end());
        next_layer.resize(unique(next_layer.begin(), next_layer.end()) - next_layer.begin());
        if (next_layer.size() > mem_size) {
            nth_element(next_layer.begin(), next_layer.begin() + mem_size, next_layer.end(), [&](const auto &elem1, const auto &elem2) {
                return Weight(elem1) < Weight(elem2);
            });
            next_layer.resize(mem_size);
        }
        layer = next_layer;
        if (find(layer.begin(), layer.end(), zeros) != layer.end()) {
            cout << layer_id + 1 << endl;
        }
    }
}
