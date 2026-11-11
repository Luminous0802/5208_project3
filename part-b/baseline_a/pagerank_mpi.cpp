#include <mpi.h>
#include <algorithm>
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

    MPI_Init(&argc, &argv);
    int mpi_rank = 0;
    int mpi_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    std::vector<int> counts(mpi_size);
    std::vector<int> displs(mpi_size);
    for (int p = 0; p < mpi_size; ++p) {
        displs[p] = static_cast<long long>(p) * N / mpi_size;
        int end = static_cast<long long>(p + 1) * N / mpi_size;
        counts[p] = end - displs[p];
    }
    const int local_start = displs[mpi_rank];
    const int local_count = counts[mpi_rank];
    const int first_edge = row_ptr[local_start];
    const int last_edge = row_ptr[local_start + local_count];
    std::vector<int> local_row_ptr(local_count + 1);
    for (int i = 0; i <= local_count; ++i) {
        local_row_ptr[i] = row_ptr[local_start + i] - first_edge;
    }
    std::vector<int> local_col_idx(col_idx.begin() + first_edge,
                                   col_idx.begin() + last_edge);
    std::vector<int>().swap(row_ptr);
    std::vector<int>().swap(col_idx);
    std::vector<int>().swap(indegree);
    std::vector<int>().swap(next);
    std::vector<std::pair<int, int>>().swap(edges);

    const double damping = 0.85;
    std::vector<double> rank(N, 1.0 / N);
    std::vector<double> next_rank(N, 0.0);
    std::vector<double> local_new(local_count, 0.0);
    double residual = 0.0;
    int completed = 0;
    double compute_s = 0.0;
    double allreduce_s = 0.0;
    double allgatherv_s = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (int iteration = 0; iteration < K; ++iteration) {
        double phase_start = MPI_Wtime();
        double local_dangling_mass = 0.0;
        for (int u = local_start; u < local_start + local_count; ++u) {
            if (outdegree[u] == 0) {
                local_dangling_mass += rank[u];
            }
        }
        compute_s += MPI_Wtime() - phase_start;
        double dangling_mass = 0.0;
        phase_start = MPI_Wtime();
        MPI_Allreduce(&local_dangling_mass, &dangling_mass, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        allreduce_s += MPI_Wtime() - phase_start;

        phase_start = MPI_Wtime();
        for (int i = 0; i < local_count; ++i) {
            double incoming = 0.0;
            for (int p = local_row_ptr[i]; p < local_row_ptr[i + 1]; ++p) {
                const int u = local_col_idx[p];
                incoming += rank[u] / outdegree[u];
            }
            local_new[i] = (1.0 - damping) / N
                         + damping * (incoming + dangling_mass / N);
        }

        compute_s += MPI_Wtime() - phase_start;
        phase_start = MPI_Wtime();
        MPI_Allgatherv(local_new.empty() ? nullptr : local_new.data(),
                       local_count, MPI_DOUBLE, next_rank.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        allgatherv_s += MPI_Wtime() - phase_start;

        phase_start = MPI_Wtime();
        double local_residual = 0.0;
        for (int i = 0; i < local_count; ++i) {
            const int v = local_start + i;
            local_residual += std::abs(next_rank[v] - rank[v]);
        }
        compute_s += MPI_Wtime() - phase_start;
        phase_start = MPI_Wtime();
        MPI_Allreduce(&local_residual, &residual, 1, MPI_DOUBLE,
                      MPI_SUM, MPI_COMM_WORLD);
        allreduce_s += MPI_Wtime() - phase_start;
        rank.swap(next_rank);
        completed = iteration + 1;
        if (check_convergence && residual < tol) {
            break;
        }
    }
    const double local_elapsed = MPI_Wtime() - start;
    double iteration_s = 0.0;
    MPI_Reduce(&local_elapsed, &iteration_s, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    double max_compute_s = 0.0;
    double max_allreduce_s = 0.0;
    double max_allgatherv_s = 0.0;
    MPI_Reduce(&compute_s, &max_compute_s, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&allreduce_s, &max_allreduce_s, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&allgatherv_s, &max_allgatherv_s, 1, MPI_DOUBLE, MPI_MAX,
               0, MPI_COMM_WORLD);
    const int local_edges = local_row_ptr.back();
    int min_local_edges = 0;
    int max_local_edges = 0;
    MPI_Reduce(&local_edges, &min_local_edges, 1, MPI_INT, MPI_MIN,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&local_edges, &max_local_edges, 1, MPI_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    if (mpi_rank == 0) {
        double rank_sum = 0.0;
        for (double value : rank) rank_sum += value;
        std::cout << std::fixed << std::setprecision(10);
        std::cout << "ranks: " << mpi_size << '\n';
        std::cout << "iterations: " << completed << '\n';
        std::cout << "iteration_s: " << iteration_s << '\n';
        std::cout << "compute_s: " << max_compute_s << '\n';
        std::cout << "allreduce_s: " << max_allreduce_s << '\n';
        std::cout << "allgatherv_s: " << max_allgatherv_s << '\n';
        std::cout << "local_edges_min: " << min_local_edges << '\n';
        std::cout << "local_edges_max: " << max_local_edges << '\n';
        std::cout << "edge_imbalance: "
                  << (E > 0 ? static_cast<double>(max_local_edges) * mpi_size / E
                            : 0.0) << '\n';
        std::cout << "rank_sum: " << rank_sum << '\n';
        std::cout << "residual: " << residual << '\n';
        if (N <= 20) {
            for (int v = 0; v < N; ++v) {
                std::cout << "rank " << v << ": " << rank[v] << '\n';
            }
        }
    }
    MPI_Finalize();
    return 0;
}
