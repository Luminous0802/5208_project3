#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: ./validate_reference GRAPH_FILE K OUTPUT_FILE\n";
        return 1;
    }

    const int iterations = std::stoi(argv[2]);
    if (iterations <= 0) {
        std::cerr << "K must be positive\n";
        return 1;
    }

    std::ifstream input(argv[1]);
    int vertex_count = 0;
    int edge_count = 0;
    if (!(input >> vertex_count >> edge_count) ||
        vertex_count <= 0 || edge_count < 0) {
        std::cerr << "Invalid graph header\n";
        return 1;
    }

    std::vector<std::pair<int, int>> edges;
    edges.reserve(edge_count);
    std::vector<int> outdegree(vertex_count, 0);
    std::unordered_set<std::uint64_t> seen;

    for (int i = 0; i < edge_count; ++i) {
        int source = 0;
        int target = 0;
        if (!(input >> source >> target) ||
            source < 0 || source >= vertex_count ||
            target < 0 || target >= vertex_count) {
            std::cerr << "Invalid edge " << i << '\n';
            return 1;
        }
        const std::uint64_t key =
            static_cast<std::uint64_t>(source) * vertex_count + target;
        if (!seen.insert(key).second) {
            std::cerr << "Duplicate edge\n";
            return 1;
        }
        edges.emplace_back(source, target);
        ++outdegree[source];
    }
    std::string extra;
    if (input >> extra) {
        std::cerr << "Extra data after declared edges\n";
        return 1;
    }

    const double damping = 0.85;
    std::vector<double> rank(vertex_count, 1.0 / vertex_count);
    std::vector<double> updated(vertex_count);
    double residual = 0.0;

    for (int iteration = 0; iteration < iterations; ++iteration) {
        double dangling_mass = 0.0;
        for (int vertex = 0; vertex < vertex_count; ++vertex) {
            if (outdegree[vertex] == 0) {
                dangling_mass += rank[vertex];
            }
        }

        const double base = (1.0 - damping) / vertex_count
                          + damping * dangling_mass / vertex_count;
        std::fill(updated.begin(), updated.end(), base);

        for (const auto& edge : edges) {
            const int source = edge.first;
            const int target = edge.second;
            updated[target] += damping * rank[source] / outdegree[source];
        }

        residual = 0.0;
        for (int vertex = 0; vertex < vertex_count; ++vertex) {
            residual += std::abs(updated[vertex] - rank[vertex]);
        }
        rank.swap(updated);
    }

    std::ofstream output(argv[3]);
    if (!output) {
        std::cerr << "Cannot open rank output file\n";
        return 1;
    }
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (double value : rank) {
        output << value << '\n';
    }
    output.close();
    if (!output) {
        std::cerr << "Failed while writing rank output file\n";
        return 1;
    }

    double rank_sum = 0.0;
    for (double value : rank) {
        rank_sum += value;
    }
    std::cout << std::setprecision(17);
    std::cout << "vertices: " << vertex_count << '\n';
    std::cout << "iterations: " << iterations << '\n';
    std::cout << "rank_sum: " << rank_sum << '\n';
    std::cout << "residual: " << residual << '\n';
    return 0;
}
