#pragma once
// Symmetric Lanczos with optional two-pass full reorthogonalization.
// Core and projected eigensolver are handwritten; no BLAS/LAPACK/graph libraries.
#include "graph.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#ifdef LANCZOS_MPI
#include <mpi.h>
#include <omp.h>
#endif

struct World {
    int rank=0,size=1,threads=1;
    double dot_compute=0,vector_compute=0,spmv_compute=0,allreduce=0,allgatherv=0;
    static double now(){
#ifdef LANCZOS_MPI
        return MPI_Wtime();
#else
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
    }
    void barrier(){
#ifdef LANCZOS_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    void sum(std::vector<double>& a){
#ifdef LANCZOS_MPI
        double t=now();
        MPI_Allreduce(MPI_IN_PLACE,a.data(),int(a.size()),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        allreduce+=now()-t;
#else
        (void)a;
#endif
    }
    double maximum(double x){
#ifdef LANCZOS_MPI
        double y=0;MPI_Allreduce(&x,&y,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);return y;
#else
        return x;
#endif
    }
    double dot(const std::vector<double>& a,const std::vector<double>& b){
        double t=now(),s=0;
#ifdef LANCZOS_MPI
        #pragma omp parallel for reduction(+:s) schedule(static)
#endif
        for(int i=0;i<int(a.size());++i)s+=a[i]*b[i];
        dot_compute+=now()-t;
        std::vector<double> v(1,s);sum(v);return v[0];
    }
    void gather(const std::vector<double>& local,std::vector<double>& global,
                const std::vector<int>& counts,const std::vector<int>& displs){
#ifdef LANCZOS_MPI
        double t=now();MPI_Allgatherv(local.empty()?nullptr:local.data(),int(local.size()),MPI_DOUBLE,
            global.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);allgatherv+=now()-t;
#else
        (void)counts;(void)displs;global=local;
#endif
    }
};
struct Symmetric {
    int n=0,input_e=0,undirected_e=0;
    std::vector<int> row,col,degree;
    std::vector<double> weight;
    explicit Symmetric(const std::string& path){
        auto g=pr::load(path);n=g.n;input_e=g.e;
        std::vector<std::pair<int,int>> edges;edges.reserve(g.e);
        for(int v=0;v<n;++v)for(int j=g.row[v];j<g.row[v+1];++j)
            edges.emplace_back(std::min(v,g.col[j]),std::max(v,g.col[j]));
        std::sort(edges.begin(),edges.end());edges.erase(std::unique(edges.begin(),edges.end()),edges.end());
        undirected_e=int(edges.size());degree.assign(n,0);
        for(auto e:edges){++degree[e.first];if(e.first!=e.second)++degree[e.second];}
        int64_t arcs=std::accumulate(degree.begin(),degree.end(),int64_t(0));
        if(arcs>INT_MAX)throw std::runtime_error("symmetrized CSR exceeds INT_MAX entries");
        row.assign(n+1,0);for(int v=0;v<n;++v)row[v+1]=row[v]+degree[v];
        col.resize(size_t(arcs));auto next=row;
        for(auto e:edges){col[next[e.first]++]=e.second;if(e.first!=e.second)col[next[e.second]++]=e.first;}
        weight.resize(col.size());
        for(int v=0;v<n;++v){
            std::sort(col.begin()+row[v],col.begin()+row[v+1]);
            for(int j=row[v];j<row[v+1];++j)weight[j]=1.0/std::sqrt(double(degree[v])*degree[col[j]]);
        }
    }
};
// Cyclic-free maximum-pivot Jacobi rotations for the tiny projected symmetric matrix.
struct Eigen {std::vector<double> values,vectors;int n=0;};
Eigen jacobi(std::vector<double> a,int n){
    Eigen e;e.n=n;e.vectors.assign(size_t(n)*n,0);
    for(int i=0;i<n;++i)e.vectors[size_t(i)*n+i]=1;
    bool converged=false;
    for(int iter=0;iter<std::max(1,100*n*n);++iter){
        int p=0,q=0;double biggest=0,scale=0;
        for(int i=0;i<n;++i){scale=std::max(scale,std::abs(a[size_t(i)*n+i]));
            for(int j=i+1;j<n;++j)if(std::abs(a[size_t(i)*n+j])>biggest){biggest=std::abs(a[size_t(i)*n+j]);p=i;q=j;}}
        if(biggest<=1e-14*std::max(1.0,scale)){converged=true;break;}
        double apq=a[size_t(p)*n+q],tau=(a[size_t(q)*n+q]-a[size_t(p)*n+p])/(2*apq);
        double t=std::copysign(1.0,tau)/(std::abs(tau)+std::hypot(1.0,tau));
        double cs=1/std::sqrt(1+t*t),sn=t*cs;
        a[size_t(p)*n+p]-=t*apq;a[size_t(q)*n+q]+=t*apq;
        a[size_t(p)*n+q]=a[size_t(q)*n+p]=0;
        for(int k=0;k<n;++k){
            if(k!=p&&k!=q){double x=a[size_t(k)*n+p],y=a[size_t(k)*n+q];
                a[size_t(k)*n+p]=a[size_t(p)*n+k]=cs*x-sn*y;
                a[size_t(k)*n+q]=a[size_t(q)*n+k]=sn*x+cs*y;}
            double x=e.vectors[size_t(k)*n+p],y=e.vectors[size_t(k)*n+q];
            e.vectors[size_t(k)*n+p]=cs*x-sn*y;e.vectors[size_t(k)*n+q]=sn*x+cs*y;
        }
    }
    if(!converged)throw std::runtime_error("projected Jacobi eigensolver did not converge");
    e.values.resize(n);for(int i=0;i<n;++i)e.values[i]=a[size_t(i)*n+i];return e;
}
Eigen projected(const std::vector<double>& alpha,const std::vector<double>& beta,int m){
    std::vector<double> a(size_t(m)*m,0);
    for(int i=0;i<m;++i){a[size_t(i)*m+i]=alpha[i];if(i+1<m)a[size_t(i)*m+i+1]=a[size_t(i+1)*m+i]=beta[i];}
    return jacobi(std::move(a),m);
}
uint64_t splitmix(uint64_t x){x+=UINT64_C(0x9e3779b97f4a7c15);x=(x^(x>>30))*UINT64_C(0xbf58476d1ce4e5b9);x=(x^(x>>27))*UINT64_C(0x94d049bb133111eb);return x^(x>>31);}
