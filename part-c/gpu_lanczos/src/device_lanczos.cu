#include "device_lanczos.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {
void check(cudaError_t code,const char* what) {
    if(code!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(code));
}
double wall(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
template<class T> struct Buffer {
    T* ptr=nullptr;
    explicit Buffer(std::size_t n){check(cudaMalloc(reinterpret_cast<void**>(&ptr),std::max<std::size_t>(1,n)*sizeof(T)),"cudaMalloc");}
    ~Buffer(){if(ptr)cudaFree(ptr);}
    Buffer(const Buffer&)=delete;
};
struct Events {
    cudaEvent_t a=nullptr,b=nullptr;
    Events(){check(cudaEventCreate(&a),"event begin");try{check(cudaEventCreate(&b),"event end");}catch(...){cudaEventDestroy(a);throw;}}
    ~Events(){if(a)cudaEventDestroy(a);if(b)cudaEventDestroy(b);}
};
__device__ void reduce_block(double x,double* output){
    extern __shared__ double values[];
    values[threadIdx.x]=x;__syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2){if(threadIdx.x<stride)values[threadIdx.x]+=values[threadIdx.x+stride];__syncthreads();}
    if(threadIdx.x==0)*output=values[0];
}
__global__ void multiply(int local_n,int start,const int* row,const int* col,const double* weight,const double* global,double* z){
    for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<local_n;i+=static_cast<long long>(blockDim.x)*gridDim.x){
        double sum=0;for(int j=row[i];j<row[i+1];++j)sum+=weight[j]*global[col[j]];
        z[i]=0.5*(sum+global[start+i]);
    }
}
__global__ void axpy(int n,double* target,const double* x,double alpha){
    for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<long long>(blockDim.x)*gridDim.x)target[i]-=alpha*x[i];
}
__global__ void divide(int n,double* target,const double* x,double divisor){
    for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<long long>(blockDim.x)*gridDim.x)target[i]=x[i]/divisor;
}
__global__ void partial_dot(int n,const double* a,const double* b,double* partial){
    double sum=0;
    for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<long long>(blockDim.x)*gridDim.x)sum+=a[i]*b[i];
    reduce_block(sum,partial+blockIdx.x);
}
__global__ void finish_dots(const double* partial,int blocks,double* answer){
    int k=blockIdx.x;double sum=0;
    for(int j=threadIdx.x;j<blocks;j+=blockDim.x)sum+=partial[static_cast<std::size_t>(k)*blocks+j];
    reduce_block(sum,answer+k);
}
__global__ void partial_basis_dot(int n,std::size_t stride,const double* basis,const double* x,double* partial){
    int k=blockIdx.y;double sum=0;
    for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<long long>(blockDim.x)*gridDim.x)sum+=basis[static_cast<std::size_t>(k)*stride+i]*x[i];
    reduce_block(sum,partial+static_cast<std::size_t>(k)*gridDim.x+blockIdx.x);
}
__global__ void combine_basis(int n,std::size_t stride,const double* basis,const double* coeff,int count,double* out,int subtract){
    for(long long i=static_cast<long long>(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=static_cast<long long>(blockDim.x)*gridDim.x){
        double sum=0;for(int k=0;k<count;++k)sum+=coeff[k]*basis[static_cast<std::size_t>(k)*stride+i];
        if(subtract)out[i]-=sum;else out[i]=sum;
    }
}
GpuInfo configure(){
    GpuInfo info;check(cudaGetDeviceCount(&info.visible_devices),"cudaGetDeviceCount");
    if(info.visible_devices!=1)throw std::runtime_error("Require exactly one CUDA-visible GPU; respect the scheduler's device assignment; no CPU fallback");
    // Respect the scheduler's CUDA_VISIBLE_DEVICES; use only visible ordinal 0.
    check(cudaSetDevice(0),"cudaSetDevice");cudaDeviceProp prop{};check(cudaGetDeviceProperties(&prop,0),"cudaGetDeviceProperties");
    info.name=prop.name;info.capability=std::to_string(prop.major)+"."+std::to_string(prop.minor);info.total_memory=prop.totalGlobalMem;
    const cudaUUID_t uuid=prop.uuid;std::ostringstream id;
    id<<std::hex<<std::setfill('0');for(unsigned char c:uuid.bytes)id<<std::setw(2)<<unsigned(c);info.uuid=id.str();
    check(cudaRuntimeGetVersion(&info.runtime),"runtime version");check(cudaDriverGetVersion(&info.driver),"driver version");return info;
}
}

struct DeviceLanczos::Impl {
    GpuInfo info;
    int global_n,start,n,steps,block,blocks;
    std::size_t stride,used=0;
    Buffer<int> row,col;
    Buffer<double> weight,q,previous,z,global,basis,partial,coeff,scalar;
    std::vector<std::unique_ptr<Events>> events;
    GpuTransfers transfer;
    Impl(GpuInfo device,int total,int first,int local,int max_steps,int block_size,
         const std::vector<int>& hrow,const std::vector<int>& hcol,const std::vector<double>& hweight)
        :info(std::move(device)),global_n(total),start(first),n(local),steps(max_steps),block(block_size),
         blocks(std::max(1,std::min(65535,int((static_cast<long long>(local)+block_size-1)/block_size)))),stride(std::max(1,local)),
         row(hrow.size()),col(hcol.size()),weight(hweight.size()),q(stride),previous(stride),z(stride),global(total),basis(stride*max_steps),
         partial(static_cast<std::size_t>(blocks)*max_steps),coeff(max_steps),scalar(1){
        copy_to(row.ptr,hrow.data(),hrow.size()*sizeof(int));copy_to(col.ptr,hcol.data(),hcol.size()*sizeof(int));copy_to(weight.ptr,hweight.data(),hweight.size()*sizeof(double));
        events.reserve(64+20*steps);for(int i=0;i<64+20*steps;++i)events.push_back(std::make_unique<Events>());
    }
    void copy_to(void* dst,const void* src,std::size_t bytes){if(!bytes)return;double t=wall();check(cudaMemcpy(dst,src,bytes,cudaMemcpyHostToDevice),"H2D");check(cudaDeviceSynchronize(),"H2D sync");transfer.h2d+=wall()-t;}
    void copy_from(void* dst,const void* src,std::size_t bytes){if(!bytes)return;double t=wall();check(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToHost),"D2H");transfer.d2h+=wall()-t;}
    template<class F> void launch(F&& f){
        if(used==events.size())throw std::runtime_error("CUDA event capacity exhausted");
        auto& e=*events[used++];check(cudaEventRecord(e.a),"event record begin");f();check(cudaGetLastError(),"kernel launch");check(cudaEventRecord(e.b),"event record end");
    }
    double dot(const double* a,const double* b){
        launch([&]{partial_dot<<<blocks,block,std::size_t(block)*sizeof(double)>>>(n,a,b,partial.ptr);});
        launch([&]{finish_dots<<<1,block,std::size_t(block)*sizeof(double)>>>(partial.ptr,blocks,scalar.ptr);});
        double value=0;copy_from(&value,scalar.ptr,sizeof(double));return value;
    }
    std::vector<double> dots(const double* x,int count){
        if(count<1||count>steps)throw std::runtime_error("Invalid basis count");
        launch([&]{partial_basis_dot<<<dim3(blocks,count),block,std::size_t(block)*sizeof(double)>>>(n,stride,basis.ptr,x,partial.ptr);});
        launch([&]{finish_dots<<<count,block,std::size_t(block)*sizeof(double)>>>(partial.ptr,blocks,coeff.ptr);});
        std::vector<double> answer(count);copy_from(answer.data(),coeff.ptr,count*sizeof(double));return answer;
    }
};
DeviceLanczos::DeviceLanczos(int n,int start,int local_n,int steps,int block,const std::vector<int>& row,const std::vector<int>& col,const std::vector<double>& weight){
    if(n<1||local_n<0||start<0||start+local_n>n||steps<1||steps>1000||row.size()!=std::size_t(local_n+1)||col.size()!=weight.size()||
       (block!=64&&block!=128&&block!=256&&block!=512))throw std::runtime_error("Invalid GPU Lanczos dimensions");
    p=std::make_unique<Impl>(configure(),n,start,local_n,steps,block,row,col,weight);
}
DeviceLanczos::~DeviceLanczos()=default;
const GpuInfo& DeviceLanczos::info()const{return p->info;}
void DeviceLanczos::initialize(const std::vector<double>& q){if(q.size()!=std::size_t(p->n))throw std::runtime_error("Initial vector length");p->copy_to(p->q.ptr,q.data(),q.size()*sizeof(double));check(cudaMemset(p->previous.ptr,0,p->stride*sizeof(double)),"zero previous");check(cudaMemset(p->z.ptr,0,p->stride*sizeof(double)),"zero z");synchronize();}
void DeviceLanczos::download_q(std::vector<double>& q){q.resize(p->n);p->copy_from(q.data(),p->q.ptr,q.size()*sizeof(double));}
void DeviceLanczos::upload_global(const std::vector<double>& q){if(q.size()!=std::size_t(p->global_n))throw std::runtime_error("Global vector length");p->copy_to(p->global.ptr,q.data(),q.size()*sizeof(double));}
void DeviceLanczos::store_basis(int j){if(j<0||j>=p->steps)throw std::runtime_error("Basis index");if(p->n)check(cudaMemcpy(p->basis.ptr+std::size_t(j)*p->stride,p->q.ptr,p->n*sizeof(double),cudaMemcpyDeviceToDevice),"store basis");}
void DeviceLanczos::spmv(){p->launch([&]{multiply<<<p->blocks,p->block>>>(p->n,p->start,p->row.ptr,p->col.ptr,p->weight.ptr,p->global.ptr,p->z.ptr);});}
void DeviceLanczos::subtract_previous(double b){p->launch([&]{axpy<<<p->blocks,p->block>>>(p->n,p->z.ptr,p->previous.ptr,b);});}
double DeviceLanczos::dot_qz(){return p->dot(p->q.ptr,p->z.ptr);}
double DeviceLanczos::dot_zz(){return p->dot(p->z.ptr,p->z.ptr);}
double DeviceLanczos::dot_qq(){return p->dot(p->q.ptr,p->q.ptr);}
void DeviceLanczos::subtract_q(double a){p->launch([&]{axpy<<<p->blocks,p->block>>>(p->n,p->z.ptr,p->q.ptr,a);});}
std::vector<double> DeviceLanczos::basis_dots(int count){return p->dots(p->z.ptr,count);}
std::vector<double> DeviceLanczos::gram_column(int column,int count){if(column<0||column>=count)throw std::runtime_error("Gram column");return p->dots(p->basis.ptr+std::size_t(column)*p->stride,count);}
void DeviceLanczos::subtract_basis(const std::vector<double>& coeff){p->copy_to(p->coeff.ptr,coeff.data(),coeff.size()*sizeof(double));p->launch([&]{combine_basis<<<p->blocks,p->block>>>(p->n,p->stride,p->basis.ptr,p->coeff.ptr,int(coeff.size()),p->z.ptr,1);});}
void DeviceLanczos::advance(double b){if(!std::isfinite(b)||b<=0)throw std::runtime_error("Invalid beta");if(p->n)check(cudaMemcpy(p->previous.ptr,p->q.ptr,p->n*sizeof(double),cudaMemcpyDeviceToDevice),"copy previous");p->launch([&]{divide<<<p->blocks,p->block>>>(p->n,p->q.ptr,p->z.ptr,b);});}
void DeviceLanczos::reconstruct(const std::vector<double>& coeff){p->copy_to(p->coeff.ptr,coeff.data(),coeff.size()*sizeof(double));p->launch([&]{combine_basis<<<p->blocks,p->block>>>(p->n,p->stride,p->basis.ptr,p->coeff.ptr,int(coeff.size()),p->q.ptr,0);});}
void DeviceLanczos::scale_q(double b){if(!std::isfinite(b)||b<=0)throw std::runtime_error("Invalid norm");p->launch([&]{divide<<<p->blocks,p->block>>>(p->n,p->q.ptr,p->q.ptr,b);});}
void DeviceLanczos::synchronize(){check(cudaDeviceSynchronize(),"CUDA synchronize");}
void DeviceLanczos::reset_measurement(){synchronize();p->used=0;p->transfer={};}
std::size_t DeviceLanczos::event_count()const{return p->used;}
double DeviceLanczos::event_seconds(std::size_t count){if(count>p->used)throw std::runtime_error("Event index");synchronize();double seconds=0;for(std::size_t i=0;i<count;++i){float ms=0;check(cudaEventElapsedTime(&ms,p->events[i]->a,p->events[i]->b),"event elapsed");seconds+=ms*0.001;}return seconds;}
GpuTransfers DeviceLanczos::transfers()const{return p->transfer;}
