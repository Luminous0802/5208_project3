#pragma once
#include "graph.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace pr {
inline double wall(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Options {
    std::string graph,dump,mapping="thread";
    int k=50,device=0,block=256,warmup=2;
    double tol=0;bool validate=false;
};
inline Options options(int argc,char** argv){
    if(argc<3)throw std::runtime_error("Usage: pagerank_reference|pagerank_cuda GRAPH K [TOL] [--dump FILE] [--mapping thread|warp] [--block-size 64|128|256|512] [--device ID] [--warmup K] [--validate]");
    Options c;c.graph=argv[1];c.k=positive(argv[2]);int a=3;
    if(a<argc&&std::string(argv[a]).rfind("--",0)!=0){std::string s=argv[a++];size_t end=0;c.tol=std::stod(s,&end);
        if(end!=s.size()||!std::isfinite(c.tol)||c.tol<=0)throw std::runtime_error("positive finite TOL required");}
    while(a<argc){std::string key=argv[a++];if(key=="--validate"){c.validate=true;continue;}
        if(a==argc)throw std::runtime_error("missing option value");std::string value=argv[a++];
        if(key=="--dump")c.dump=value;else if(key=="--mapping")c.mapping=value;
        else if(key=="--block-size")c.block=positive(value);
        else if(key=="--device"||key=="--warmup"){auto v=integer(value);if(v<0||v>INT_MAX)throw std::runtime_error("invalid nonnegative integer");if(key=="--device")c.device=int(v);else c.warmup=int(v);}
        else throw std::runtime_error("unknown option: "+key);
    }
    if(c.mapping!="thread"&&c.mapping!="warp")throw std::runtime_error("mapping must be thread or warp");
    if(c.block!=64&&c.block!=128&&c.block!=256&&c.block!=512)throw std::runtime_error("block-size must be 64,128,256,512");
    return c;
}
struct Result {std::vector<double> rank;int completed=0;double residual=0,seconds=0;};
inline Result reference(const Graph& g,int k,double tol){
    Result r;r.rank.assign(g.n,1.0/g.n);std::vector<double> next(g.n);double begin=wall();
    for(int it=0;it<k;++it){double mass=0;for(int u=0;u<g.n;++u)if(g.out[u]==0)mass+=r.rank[u];
        double diff=0;
        for(int v=0;v<g.n;++v){double incoming=0;
            for(int j=g.row[v];j<g.row[v+1];++j){int u=g.col[j];incoming+=r.rank[u]/g.out[u];}
            next[v]=(1.0-0.85)/g.n+0.85*(incoming+mass/g.n);diff+=std::abs(next[v]-r.rank[v]);}
        r.rank.swap(next);r.residual=diff;r.completed=it+1;if(tol>0&&diff<tol)break;
    }
    r.seconds=wall()-begin;return r;
}
inline void dump_vector(const std::string& path,const std::vector<double>& values){
    if(path.empty())return;std::ofstream out(path);if(!out)throw std::runtime_error("cannot write rank dump");out<<std::setprecision(17);
    for(size_t i=0;i<values.size();++i)out<<i<<' '<<values[i]<<'\n';
}
inline void print_rank(const std::vector<double>& values){if(values.size()<=20)for(size_t i=0;i<values.size();++i)std::cout<<"rank "<<i<<": "<<values[i]<<'\n';}
}
