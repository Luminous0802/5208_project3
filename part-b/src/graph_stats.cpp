#include "graph.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>

struct DSU {
    std::vector<int> par,size;
    explicit DSU(int n):par(n),size(n,1){std::iota(par.begin(),par.end(),0);}
    int root(int v){while(v!=par[v]){par[v]=par[par[v]];v=par[v];}return v;}
    void join(int u,int v){u=root(u);v=root(v);if(u==v)return;if(size[u]<size[v])std::swap(u,v);par[v]=u;size[u]+=size[v];}
};
int main(int argc,char** argv){
    try{
        if(argc!=4)throw std::runtime_error("Usage: graph_stats GRAPH RANKS vertex|edge (JSON to stdout)");
        auto g=pr::load(argv[1]); int p=pr::positive(argv[2]); std::string part=argv[3];
        if(part!="vertex"&&part!="edge")throw std::runtime_error("partition must be vertex or edge");
        auto b=pr::partition(g,p,part=="edge");
        DSU dsu(g.n); std::vector<int> owner(g.n),mark(g.n,-1);
        for(int r=0;r<p;++r)for(int v=b[r];v<b[r+1];++v)owner[v]=r;
        int maxin=0,maxout=0,dangling=0,isolated=0; long long self=0,cross=0;
        long double in2=0,out2=0;
        for(int v=0;v<g.n;++v){
            int deg=g.row[v+1]-g.row[v];maxin=std::max(maxin,deg);maxout=std::max(maxout,g.out[v]);
            in2+=static_cast<long double>(deg)*deg;out2+=static_cast<long double>(g.out[v])*g.out[v];
            dangling+=(g.out[v]==0);isolated+=(g.out[v]==0&&deg==0);
            for(int j=g.row[v];j<g.row[v+1];++j){int u=g.col[j];dsu.join(u,v);self+=(u==v);cross+=(owner[u]!=owner[v]);}
        }
        int components=0,largest=0;
        for(int v=0;v<g.n;++v)if(dsu.root(v)==v){++components;largest=std::max(largest,dsu.size[v]);}
        int maxedges=0;long long unique_remote_total=0;
        for(int r=0;r<p;++r)maxedges=std::max(maxedges,g.row[b[r+1]]-g.row[b[r]]);
        double avg=double(g.e)/g.n;
        std::cout<<std::setprecision(17)<<"{\n\"N\":"<<g.n<<",\n\"E\":"<<g.e
            <<",\n\"directed_density_excluding_self\":"<<(g.n>1?double(g.e-self)/(double(g.n)*(g.n-1)):0)
            <<",\n\"mean_indegree\":"<<avg<<",\n\"mean_outdegree\":"<<avg
            <<",\n\"max_indegree\":"<<maxin<<",\n\"max_outdegree\":"<<maxout
            <<",\n\"indegree_population_variance\":"<<std::max(0.0,double(in2/g.n)-avg*avg)
            <<",\n\"outdegree_population_variance\":"<<std::max(0.0,double(out2/g.n)-avg*avg)
            <<",\n\"dangling_fraction\":"<<double(dangling)/g.n
            <<",\n\"isolated_fraction\":"<<double(isolated)/g.n
            <<",\n\"weak_components\":"<<components<<",\n\"largest_weak_component_fraction\":"<<double(largest)/g.n
            <<",\n\"self_loops\":"<<self<<",\n\"partition\":\""<<part<<"\",\n\"ranks\":"<<p
            <<",\n\"edge_imbalance\":"<<(g.e?double(maxedges)*p/g.e:0)
            <<",\n\"cross_partition_edge_fraction\":"<<(g.e?double(cross)/g.e:0)<<",\n\"per_rank\":[\n";
        for(int r=0;r<p;++r){
            int remote=0;
            for(int v=b[r];v<b[r+1];++v)for(int j=g.row[v];j<g.row[v+1];++j){
                int u=g.col[j];if(owner[u]!=r&&mark[u]!=r){mark[u]=r;++remote;}
            }
            unique_remote_total+=remote;
            std::cout<<(r?",\n":"")<<"{\"rank\":"<<r<<",\"start\":"<<b[r]<<",\"end\":"<<b[r+1]
                <<",\"vertices\":"<<b[r+1]-b[r]<<",\"edges\":"<<g.row[b[r+1]]-g.row[b[r]]
                <<",\"remote_unique_sources\":"<<remote<<",\"remote_unique_sources_over_N\":"<<double(remote)/g.n<<'}';
        }
        // Denominator P*N: mean fraction of the global vector needed remotely per rank.
        std::cout<<"\n],\n\"mean_remote_unique_sources_over_N\":"<<double(unique_remote_total)/(double(p)*g.n)
            <<",\n\"allgatherv_logical_receive_bytes_per_rank_per_iteration\":"<<uint64_t(g.n)*sizeof(double)
            <<"\n}\n";
    }catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
}
