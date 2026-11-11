#pragma once
#include <algorithm>
#include <climits>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace pr {
struct Graph {
    int n = 0, e = 0;
    std::vector<int> row, col, out;
};
inline Graph load(const std::string& path) {
    std::ifstream in(path);
    long long n, e;
    if (!(in >> n >> e) || n <= 0 || n >= INT_MAX || e < 0 || e > INT_MAX)
        throw std::runtime_error("invalid header (requires 0<N<INT_MAX, 0<=E<=INT_MAX)");
    Graph g; g.n = int(n); g.e = int(e);
    g.row.assign(g.n + 1, 0); g.out.assign(g.n, 0);
    std::vector<std::pair<int,int>> edges; edges.reserve(g.e);
    for (int i = 0; i < g.e; ++i) {
        long long s, d;
        if (!(in >> s >> d) || s < 0 || s >= n || d < 0 || d >= n)
            throw std::runtime_error("invalid edge " + std::to_string(i));
        edges.emplace_back(int(s), int(d)); ++g.row[int(d)+1]; ++g.out[int(s)];
    }
    std::string extra;
    if (in >> extra) throw std::runtime_error("extra data after declared E edges");
    std::sort(edges.begin(), edges.end());
    if (std::adjacent_find(edges.begin(), edges.end()) != edges.end())
        throw std::runtime_error("duplicate edge");
    for (int v = 0; v < g.n; ++v) g.row[v+1] += g.row[v];
    auto next = g.row; g.col.resize(g.e);
    for (const auto& edge : edges) g.col[next[edge.second]++] = edge.first;
    return g;
}
// Contiguous destination ranges. Nearest indegree-prefix boundary, lower ID on ties.
// A vertex is never split. Empty partitions are legal, including P>N.
inline std::vector<int> partition(const Graph& g, int p, bool edge) {
    if (p <= 0) throw std::runtime_error("ranks must be positive");
    std::vector<int> b(p+1, 0); b[p] = g.n;
    for (int r = 1; r < p; ++r) {
        if (!edge || g.e == 0) b[r] = int(int64_t(r) * g.n / p);
        else {
            // Compare scaled integer distances: avoids rounding E*r/P early.
            int64_t target = int64_t(g.e) * r;
            int lo = 0, hi = g.n;
            while (lo < hi) {
                int mid = lo + (hi-lo)/2;
                if (int64_t(g.row[mid])*p < target) lo = mid+1; else hi = mid;
            }
            int cut = lo;
            if (cut > 0 && target-int64_t(g.row[cut-1])*p <= int64_t(g.row[cut])*p-target)
                --cut;
            b[r] = cut;
        }
    }
    return b;
}
inline long long integer(const std::string& s) {
    size_t end = 0; long long x = std::stoll(s, &end);
    if (end != s.size()) throw std::runtime_error("invalid integer: " + s);
    return x;
}
inline int positive(const std::string& s) {
    auto x = integer(s);
    if (x <= 0 || x > INT_MAX) throw std::runtime_error("expected positive int: " + s);
    return int(x);
}
}
