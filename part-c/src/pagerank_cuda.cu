// FP64 pull PageRank. Handwritten CUDA kernels and reductions; no cuBLAS/Thrust/CUB.
#include "pagerank_common.hpp"
#include <cuda_runtime.h>
#include <limits>
#include <sstream>

inline void check(cudaError_t code,const char* expr){if(code!=cudaSuccess)throw std::runtime_error(std::string(expr)+": "+cudaGetErrorString(code));}
#define CUDA_CHECK(expr) check((expr),#expr)
template<class T> struct DeviceArray {
    T* p=nullptr;explicit DeviceArray(size_t n){CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p),std::max<size_t>(n,1)*sizeof(T)));}
    ~DeviceArray(){if(p)cudaFree(p);}DeviceArray(const DeviceArray&)=delete;DeviceArray& operator=(const DeviceArray&)=delete;
};
struct Events {
    cudaEvent_t begin=nullptr,end=nullptr;
    Events(){CUDA_CHECK(cudaEventCreate(&begin));try{CUDA_CHECK(cudaEventCreate(&end));}catch(...){cudaEventDestroy(begin);throw;}}
    ~Events(){if(begin)cudaEventDestroy(begin);if(end)cudaEventDestroy(end);}
    void start(){CUDA_CHECK(cudaEventRecord(begin));}
    double finish(){CUDA_CHECK(cudaEventRecord(end));CUDA_CHECK(cudaEventSynchronize(end));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));return ms*0.001;}
};
__device__ void block_sum(double value,double* output){
    extern __shared__ double scratch[];unsigned int t=threadIdx.x;scratch[t]=value;__syncthreads();
    for(unsigned int stride=blockDim.x/2;stride>0;stride/=2){if(t<stride)scratch[t]+=scratch[t+stride];__syncthreads();}
    if(t==0)output[blockIdx.x]=scratch[0];
}
__global__ void dangling_partial(const double* rank,const int* out,int n,double* partial){
    double s=0;for(long long u=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;u<n;u+=static_cast<long long>(gridDim.x)*blockDim.x)if(out[u]==0)s+=rank[u];
    block_sum(s,partial);
}
__global__ void sum_partials(const double* partial,int n,double* scalar){
    double s=0;for(int i=threadIdx.x;i<n;i+=blockDim.x)s+=partial[i];block_sum(s,scalar);
}
__global__ void update_thread(const int* row,const int* col,const int* out,const double* rank,double* next,const double* mass,int n){
    for(long long v=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;v<n;v+=static_cast<long long>(gridDim.x)*blockDim.x){
        double incoming=0;for(int j=row[v];j<row[v+1];++j){int u=col[j];incoming+=rank[u]/out[u];}
        next[v]=(1.0-0.85)/n+0.85*(incoming+mass[0]/n);
    }
}
__global__ void update_warp(const int* row,const int* col,const int* out,const double* rank,double* next,const double* mass,int n){
    int lane=threadIdx.x&31;
    long long warp=(static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    long long warp_count=static_cast<long long>(gridDim.x)*blockDim.x/32;
    // All lanes in a warp share the same destination and loop condition.
    for(long long v=warp;v<n;v+=warp_count){
        double incoming=0;
        for(long long j=static_cast<long long>(row[v])+lane;j<row[v+1];j+=32){int u=col[j];incoming+=rank[u]/out[u];}
        for(int delta=16;delta>0;delta/=2)incoming+=__shfl_down_sync(0xffffffff,incoming,delta);
        if(lane==0)next[v]=(1.0-0.85)/n+0.85*(incoming+mass[0]/n);
    }
}
__global__ void residual_partial(const double* rank,const double* next,int n,double* partial){
    double s=0;for(long long v=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;v<n;v+=static_cast<long long>(gridDim.x)*blockDim.x)s+=fabs(next[v]-rank[v]);
    block_sum(s,partial);
}
int main(int argc,char** argv){try{
    const double app_start=pr::wall();auto c=pr::options(argc,argv);double t=pr::wall();auto g=pr::load(c.graph);double load=pr::wall()-t;
    t=pr::wall();int available=0;CUDA_CHECK(cudaGetDeviceCount(&available));
    if(c.device>=available)throw std::runtime_error("requested CUDA device is unavailable; no CPU fallback is allowed");
    CUDA_CHECK(cudaSetDevice(c.device));cudaDeviceProp prop{};CUDA_CHECK(cudaGetDeviceProperties(&prop,c.device));
    if(c.block>prop.maxThreadsPerBlock)throw std::runtime_error("block size exceeds device limit");
    if(prop.major<6)throw std::runtime_error("this implementation requires compute capability >=6.0");
    int blocks=static_cast<int>(std::min<long long>(65535,(static_cast<long long>(g.n)+c.block-1)/c.block));
    int update_blocks=c.mapping=="thread"?blocks:static_cast<int>(std::min<long long>(65535,(static_cast<long long>(g.n)*32+c.block-1)/c.block));
    DeviceArray<int> row(g.row.size()),col(g.col.size()),out(g.out.size());
    DeviceArray<double> a(g.n),b(g.n),partial(blocks),mass(1),residual(1);Events iteration_events;
    double setup=pr::wall()-t;
    std::vector<double> initial(g.n,1.0/g.n),answer(g.n);
    t=pr::wall();CUDA_CHECK(cudaMemcpy(row.p,g.row.data(),g.row.size()*sizeof(int),cudaMemcpyHostToDevice));
    if(g.e)CUDA_CHECK(cudaMemcpy(col.p,g.col.data(),g.col.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(out.p,g.out.data(),g.out.size()*sizeof(int),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(a.p,initial.data(),size_t(g.n)*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());double h2d=pr::wall()-t;
    double* old=a.p;double* next=b.p;
    auto iteration=[&](){
        dangling_partial<<<blocks,c.block,size_t(c.block)*sizeof(double)>>>(old,out.p,g.n,partial.p);CUDA_CHECK(cudaGetLastError());
        sum_partials<<<1,c.block,size_t(c.block)*sizeof(double)>>>(partial.p,blocks,mass.p);CUDA_CHECK(cudaGetLastError());
        if(c.mapping=="thread")update_thread<<<update_blocks,c.block>>>(row.p,col.p,out.p,old,next,mass.p,g.n);
        else update_warp<<<update_blocks,c.block>>>(row.p,col.p,out.p,old,next,mass.p,g.n);
        CUDA_CHECK(cudaGetLastError());
        residual_partial<<<blocks,c.block,size_t(c.block)*sizeof(double)>>>(old,next,g.n,partial.p);CUDA_CHECK(cudaGetLastError());
        sum_partials<<<1,c.block,size_t(c.block)*sizeof(double)>>>(partial.p,blocks,residual.p);CUDA_CHECK(cudaGetLastError());std::swap(old,next);
    };
    t=pr::wall();for(int i=0;i<c.warmup;++i)iteration();CUDA_CHECK(cudaDeviceSynchronize());
    old=a.p;next=b.p;CUDA_CHECK(cudaMemcpy(old,initial.data(),size_t(g.n)*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize()); // Keep reset transfer outside the measured loop.
    double warmup=pr::wall()-t;
    int completed=0;double diff=0;t=pr::wall();iteration_events.start();
    for(int i=0;i<c.k;++i){iteration();completed=i+1;
        if(c.tol>0){CUDA_CHECK(cudaMemcpy(&diff,residual.p,sizeof(double),cudaMemcpyDeviceToHost));if(diff<c.tol)break;}}
    double timeline=iteration_events.finish();double elapsed=pr::wall()-t;
    t=pr::wall();CUDA_CHECK(cudaMemcpy(answer.data(),old,size_t(g.n)*sizeof(double),cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&diff,residual.p,sizeof(double),cudaMemcpyDeviceToHost));double d2h=pr::wall()-t;
    const double end_to_end=pr::wall()-app_start;
    double rank_sum=std::accumulate(answer.begin(),answer.end(),0.0);
    for(double value:answer)if(!std::isfinite(value)||value<0)throw std::runtime_error("invalid GPU rank value");
    if(std::abs(rank_sum-1)>1e-8)throw std::runtime_error("GPU rank sum validation failed");
    double l1=-1;
    if(c.validate){auto reference=pr::reference(g,c.k,c.tol);l1=0;for(int v=0;v<g.n;++v)l1+=std::abs(answer[v]-reference.rank[v]);
        if(l1>1e-8||reference.completed!=completed)throw std::runtime_error("CPU/GPU full-vector or convergence validation failed");}
    int runtime=0,driver=0;CUDA_CHECK(cudaRuntimeGetVersion(&runtime));CUDA_CHECK(cudaDriverGetVersion(&driver));
    std::cout<<std::setprecision(17)<<"backend: cuda\nprecision: fp64\nN: "<<g.n<<"\nE: "<<g.e<<"\ngpu_name: "<<prop.name
      <<"\ndevice_ordinal: "<<c.device<<"\ncompute_capability: "<<prop.major<<'.'<<prop.minor<<"\ngpu_global_memory_bytes: "<<prop.totalGlobalMem
      <<"\ncuda_runtime_version: "<<runtime<<"\ncuda_driver_version: "<<driver<<"\nmapping: "<<c.mapping<<"\nblock_size: "<<c.block
      <<"\niterations: "<<completed<<"\niteration_s: "<<elapsed<<"\ngpu_timeline_s: "<<timeline<<"\nload_s: "<<load<<"\ngpu_setup_s: "<<setup
      <<"\nh2d_s: "<<h2d<<"\nd2h_s: "<<d2h<<"\nwarmup_s: "<<warmup<<"\ntransfer_inclusive_s: "<<h2d+elapsed+d2h
      <<"\nend_to_end_s: "<<end_to_end<<"\nrank_sum: "<<rank_sum<<"\nresidual: "<<diff<<"\nvalidation_L1: "<<l1<<'\n';
    pr::print_rank(answer);pr::dump_vector(c.dump,answer);return 0;
}catch(const std::exception& e){std::cerr<<"CUDA PageRank: "<<e.what()<<'\n';return 1;}}
