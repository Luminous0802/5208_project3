#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: ./make_pilot_graph N E SEED OUTPUT_FILE\n";
        return 1;
    }

    const std::uint64_t n_input = std::stoull(argv[1]);
    const std::uint64_t edge_count = std::stoull(argv[2]);
    const std::uint64_t seed = std::stoull(argv[3]);
    if (n_input < 2 || n_input > static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
        edge_count > n_input * (n_input - 1)) {
        std::cerr << "Invalid N or E\n";
        return 1;
    }

    const int N = static_cast<int>(n_input);
    std::ofstream output(argv[4]);
    if (!output) {
        std::cerr << "Cannot open output file\n";
        return 1;
    }

    std::mt19937_64 rng(seed);
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(static_cast<std::size_t>(edge_count * 2));
    output << N << ' ' << edge_count << '\n';
    for (std::uint64_t written = 0; written < edge_count;) {
        const int src = static_cast<int>(rng() % n_input);
        const int dst = static_cast<int>(rng() % n_input);
        if (src == dst) continue;

        const std::uint64_t key = static_cast<std::uint64_t>(src) * n_input + dst;
        if (seen.insert(key).second) {
            output << src << ' ' << dst << '\n';
            ++written;
        }
    }
    output.close();
    if (!output) {
        std::cerr << "Failed while writing output file\n";
        return 1;
    }

    std::cout << "N=" << N << " E=" << edge_count << " seed=" << seed << '\n';
    return 0;
}
