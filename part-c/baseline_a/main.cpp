#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: ./pagerank GRAPH_FILE K [TOL]\n";
        return 1;
    }

    int K = std::stoi(argv[2]);
    const bool check_convergence = (argc == 4);
    const double tol = check_convergence ? std::stod(argv[3]) : 0.0;
    if (check_convergence && (!std::isfinite(tol) || tol <= 0.0)) {
        std::cerr << "TOL must be positive and finite\n";
        return 1;
    }
    if (K <= 0) {
        std::cerr << "K must be positive\n";
        return 1;
    }

    std::ifstream input(argv[1]);
    int N, E;
    if (!(input >> N >> E) || N <= 0 || E < 0) {
        std::cerr << "Invalid graph header\n";
        return 1;
    }

    std::vector<int> indegree(N, 0);
    std::vector<int> outdegree(N, 0);
    std::vector<std::pair<int, int>> edges;
    edges.reserve(E);

    for (int i = 0; i < E; ++i) {
        int src, dst;
        if (!(input >> src >> dst) ||
            src < 0 || src >= N || dst < 0 || dst >= N) {
            std::cerr << "Invalid edge " << i << '\n';
            return 1;
        }
        edges.emplace_back(src, dst);
        ++outdegree[src];
        ++indegree[dst];
    }

    std::sort(edges.begin(), edges.end());
    if (std::adjacent_find(edges.begin(), edges.end()) != edges.end()) {
        std::cerr << "Duplicate edge\n";
        return 1;
    }

    std::vector<int> row_ptr(N + 1, 0);
    for (int v = 0; v < N; ++v) {
        row_ptr[v + 1] = row_ptr[v] + indegree[v];
    }

    std::vector<int> col_idx(E);
    std::vector<int> next = row_ptr;
    for (const auto& edge : edges) {
        col_idx[next[edge.second]++] = edge.first;
    }

    const double damping = 0.85;
    std::vector<double> rank(N, 1.0 / N);
    std::vector<double> new_rank(N, 0.0);
    double residual = 0.0;
    int completed = 0;
    const auto start = std::chrono::steady_clock::now();

    for (int iteration = 0; iteration < K; ++iteration) {
        double dangling_mass = 0.0;
        for (int u = 0; u < N; ++u) {
            if (outdegree[u] == 0) {
                dangling_mass += rank[u];
            }
        }

        residual = 0.0;
        for (int v = 0; v < N; ++v) {
            double incoming = 0.0;
            for (int p = row_ptr[v]; p < row_ptr[v + 1]; ++p) {
                int u = col_idx[p];
                incoming += rank[u] / outdegree[u];
            }

            new_rank[v] = (1.0 - damping) / N
                        + damping * (incoming + dangling_mass / N);
            residual += std::abs(new_rank[v] - rank[v]);
        }

        rank.swap(new_rank);
        completed = iteration + 1;
        if (check_convergence && residual < tol) {
            break;
        }
    }
    const auto end = std::chrono::steady_clock::now();
    const double iteration_s =
        std::chrono::duration<double>(end - start).count();
    double rank_sum = 0.0;
    for (double value : rank) rank_sum += value;

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "iterations: " << completed << '\n';
    std::cout << "iteration_s: " << iteration_s << '\n';
    std::cout << "rank_sum: " << rank_sum << '\n';
    std::cout << "residual: " << residual << '\n';
    if (N <= 20) {
        for (int v = 0; v < N; ++v) {
            std::cout << "rank " << v << ": " << rank[v] << '\n';
        }
    }
}
