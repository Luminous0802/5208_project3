#pragma once
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct GpuInfo {
    std::string name, uuid, capability;
    int visible_devices=0, runtime=0, driver=0;
    std::size_t total_memory=0;
};
struct GpuTransfers { double h2d=0, d2h=0; };

// CUDA-only implementation: MPI always receives ordinary host buffers.
class DeviceLanczos {
    struct Impl;
    std::unique_ptr<Impl> p;
public:
    DeviceLanczos(int n, int start, int local_n, int steps, int block,
                  const std::vector<int>& row, const std::vector<int>& col,
                  const std::vector<double>& weight);
    ~DeviceLanczos();
    DeviceLanczos(const DeviceLanczos&)=delete;
    DeviceLanczos& operator=(const DeviceLanczos&)=delete;
    const GpuInfo& info() const;
    void initialize(const std::vector<double>& q);
    void download_q(std::vector<double>& q);
    void upload_global(const std::vector<double>& global);
    void store_basis(int j);
    void spmv();
    void subtract_previous(double beta);
    double dot_qz();
    double dot_zz();
    double dot_qq();
    void subtract_q(double alpha);
    std::vector<double> basis_dots(int count);
    std::vector<double> gram_column(int column, int count);
    void subtract_basis(const std::vector<double>& coeff);
    void advance(double beta);
    void reconstruct(const std::vector<double>& coeff);
    void scale_q(double divisor);
    void synchronize();
    void reset_measurement();
    std::size_t event_count() const;
    double event_seconds(std::size_t count);
    GpuTransfers transfers() const;
};
