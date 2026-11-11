#include "pagerank_common.hpp"
int main(int argc,char** argv){try{
    double begin=pr::wall();auto c=pr::options(argc,argv);double t=pr::wall();auto g=pr::load(c.graph);double load=pr::wall()-t;
    auto result=pr::reference(g,c.k,c.tol);double e2e=pr::wall()-begin;
    std::cout<<std::setprecision(17)<<"backend: cpu_serial\nprecision: fp64\nN: "<<g.n<<"\nE: "<<g.e
      <<"\niterations: "<<result.completed<<"\niteration_s: "<<result.seconds<<"\nload_s: "<<load<<"\nend_to_end_s: "<<e2e
      <<"\nrank_sum: "<<std::accumulate(result.rank.begin(),result.rank.end(),0.0)<<"\nresidual: "<<result.residual<<'\n';
    pr::print_rank(result.rank);pr::dump_vector(c.dump,result.rank);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
