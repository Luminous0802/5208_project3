#include "graph.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <unordered_set>

struct RNG {
    std::mt19937_64 engine;
    explicit RNG(uint64_t seed):engine(seed){}
    uint64_t bounded(uint64_t n) {
        const uint64_t threshold = -n % n;
        uint64_t x; do {x=engine();} while(x<threshold);
        return x%n;
    }
    double unit() {return (engine()>>11)*0x1.0p-53;}
};
int main(int argc,char** argv) {
    try {
        if(argc<6) throw std::runtime_error(
            "Usage: make_graph uniform|hub|community N E SEED OUTPUT "
            "[--hub-fraction F] [--hub-prob P] [--communities C] [--mix P] [--shuffle 0|1]");
        std::string model=argv[1],path=argv[5]; int n=pr::positive(argv[2]);
        long long ee=pr::integer(argv[3]),seed=pr::integer(argv[4]);
        if(n==INT_MAX || ee<0 || ee>INT_MAX || ee>int64_t(n)*(n-1) || seed<0)
            throw std::runtime_error("invalid N/E/seed (simple directed graph; no self loops)");
        int e=int(ee),communities=4,shuffle=0; double hf=0.01,hp=0.8,mix=0.1;
        for(int a=6;a<argc;) {
            std::string key=argv[a++]; if(a==argc) throw std::runtime_error("missing option value");
            std::string val=argv[a++]; size_t end=0;
            if(key=="--communities") communities=pr::positive(val);
            else if(key=="--shuffle") { auto x=pr::integer(val); if(x!=0&&x!=1) throw std::runtime_error("shuffle 0 or 1"); shuffle=int(x); }
            else {
                double x=std::stod(val,&end);
                if(end!=val.size() || !std::isfinite(x)) throw std::runtime_error("invalid probability");
                if(key=="--hub-fraction") hf=x;
                else if(key=="--hub-prob") hp=x;
                else if(key=="--mix") mix=x;
                else throw std::runtime_error("unknown option: "+key);
            }
        }
        if(model!="uniform"&&model!="hub"&&model!="community") throw std::runtime_error("unknown model");
        if(hf<=0||hf>1||hp<0||hp>=1||mix<0||mix>1||(model=="community"&&communities>n))
            throw std::runtime_error("invalid parameters; hub-prob must be <1 to keep full support");
        int hubs=std::max(1,int(n*hf));
        std::vector<int> bounds(communities+1),group(n);
        int64_t within_capacity=0;
        for(int c=0;c<=communities;++c) bounds[c]=int(int64_t(c)*n/communities);
        for(int c=0;c<communities;++c) {
            int size=bounds[c+1]-bounds[c]; within_capacity+=int64_t(size)*(size-1);
            for(int v=bounds[c];v<bounds[c+1];++v) group[v]=c;
        }
        if(model=="community") {
            if(communities<2 && mix>0) throw std::runtime_error("external mixing needs >=2 communities");
            int64_t cap=mix==0?within_capacity:(mix==1?int64_t(n)*(n-1)-within_capacity:int64_t(n)*(n-1));
            if(e>cap) throw std::runtime_error("E exceeds support of community parameters");
            if(mix<1) for(int c=0;c<communities;++c)
                if(bounds[c+1]-bounds[c]<2) throw std::runtime_error("within-community sampling needs >=2 vertices/group");
        }
        RNG rng{uint64_t(seed)};
        std::unordered_set<uint64_t> seen; seen.reserve(size_t(e));
        std::vector<std::pair<int,int>> edges; edges.reserve(e);
        uint64_t attempts=0,limit=std::max<uint64_t>(1000000,uint64_t(e)*1000);
        while(int(edges.size())<e) {
            if(++attempts>limit) throw std::runtime_error("rejection limit: reduce density or hub concentration");
            int s=int(rng.bounded(n)),d=0;
            if(model=="uniform") d=int(rng.bounded(n));
            else if(model=="hub") d=int(rng.bounded(rng.unit()<hp?hubs:n));
            else {
                int c=group[s],size=bounds[c+1]-bounds[c];
                if(rng.unit()<mix) {
                    int pick=int(rng.bounded(n-size)); d=pick<bounds[c]?pick:pick+size;
                } else d=bounds[c]+int(rng.bounded(size));
            }
            if(s==d) continue;
            uint64_t key=uint64_t(s)*uint64_t(n)+uint64_t(d);
            if(seen.insert(key).second) edges.emplace_back(s,d);
        }
        int accepted_external=0,accepted_hub=0;
        for(const auto& edge:edges) {
            accepted_external+=(group[edge.first]!=group[edge.second]);
            accepted_hub+=(edge.second<hubs);
        }
        // Relabel only AFTER topology generation, using an independent RNG stream.
        // Thus ordered/shuffled with the same seed have exactly isomorphic topology.
        if(shuffle) {
            RNG labels{uint64_t(seed)^UINT64_C(0x9e3779b97f4a7c15)};
            std::vector<int> perm(n); for(int v=0;v<n;++v) perm[v]=v;
            for(int i=n-1;i>0;--i) std::swap(perm[i],perm[labels.bounded(i+1)]);
            for(auto& edge:edges) {edge.first=perm[edge.first];edge.second=perm[edge.second];}
            std::ofstream map(path+".labels.tsv"); if(!map) throw std::runtime_error("cannot write labels");
            map<<"old_id\tnew_id\tcommunity\n";
            for(int v=0;v<n;++v) map<<v<<'\t'<<perm[v]<<'\t'<<group[v]<<'\n';
        }
        std::sort(edges.begin(),edges.end());
        std::ofstream out(path); if(!out) throw std::runtime_error("cannot create graph");
        out<<n<<' '<<e<<'\n'; for(auto edge:edges) out<<edge.first<<' '<<edge.second<<'\n';
        out.close(); if(!out) throw std::runtime_error("graph write failed");
        std::ofstream meta(path+".generation.json"); if(!meta) throw std::runtime_error("cannot write metadata");
        meta<<std::setprecision(17)<<"{\n  \"generator\": \"exact-e-rejection-v1\",\n  \"model\": \""<<model
            <<"\",\n  \"N\": "<<n<<",\n  \"E\": "<<e<<",\n  \"seed\": "<<seed
            <<",\n  \"hub_fraction\": "<<hf<<",\n  \"hub_count\": "<<hubs<<",\n  \"hub_proposal_probability\": "<<hp
            <<",\n  \"communities\": "<<communities<<",\n  \"external_proposal_probability\": "<<mix
            <<",\n  \"actual_external_community_fraction\": "<<(e?double(accepted_external)/e:0)
            <<",\n  \"actual_destination_hub_fraction\": "<<(e?double(accepted_hub)/e:0)
            <<",\n  \"shuffled\": "<<(shuffle?"true":"false")<<",\n  \"attempts\": "<<attempts<<"\n}\n";
        std::cout<<"generated "<<path<<" N="<<n<<" E="<<e<<" attempts="<<attempts<<'\n';
    }catch(const std::exception& ex){std::cerr<<ex.what()<<'\n';return 1;}
}
