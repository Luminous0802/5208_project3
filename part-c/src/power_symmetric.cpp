#include "symmetric.hpp"
// Shifted power iteration on B=(I+S)/2 by default. Both methods can use the same B.
int main(int argc,char** argv){World w;
#ifdef LANCZOS_MPI
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&w.rank);MPI_Comm_size(MPI_COMM_WORLD,&w.size);
    if(provided<MPI_THREAD_FUNNELED){MPI_Abort(MPI_COMM_WORLD,1);return 1;}omp_set_dynamic(0);
    #pragma omp parallel
    {
        #pragma omp single
        w.threads=omp_get_num_threads();
    }
#endif
    const double app=World::now();try{
        if(argc<3)throw std::runtime_error("Usage: power_symmetric[_mpi] GRAPH STEPS [--operator normalized|lazy] [--seed INT] [--tol TOL] [--dump PREFIX] [--history CSV]");
        std::string graph=argv[1],kind="lazy",dump,history;int requested=pr::positive(argv[2]),seed=20261029;double tol=0;
        for(int a=3;a<argc;){std::string key=argv[a++];if(a==argc)throw std::runtime_error("missing option value");std::string value=argv[a++];
            if(key=="--operator"){if(value!="normalized"&&value!="lazy")throw std::runtime_error("operator must be normalized or lazy");kind=value;}
            else if(key=="--seed"){auto s=pr::integer(value);if(s<0||s>INT_MAX)throw std::runtime_error("invalid seed");seed=int(s);}
            else if(key=="--tol"){size_t end=0;tol=std::stod(value,&end);if(end!=value.size()||!std::isfinite(tol)||tol<=0)throw std::runtime_error("invalid tolerance");}
            else if(key=="--dump")dump=value;else if(key=="--history")history=value;else throw std::runtime_error("unknown option");
        }
        double t=World::now();Symmetric g(graph);double load=World::now()-t;
        std::vector<int> count(w.size),disp(w.size);
        for(int r=0;r<w.size;++r){disp[r]=int(int64_t(r)*g.n/w.size);count[r]=int(int64_t(r+1)*g.n/w.size)-disp[r];}
        int start=disp[w.rank],local=count[w.rank];int first=g.row[start],last=g.row[start+local];
        std::vector<int> row(local+1),col(g.col.begin()+first,g.col.begin()+last);std::vector<double> weight(g.weight.begin()+first,g.weight.begin()+last);
        for(int i=0;i<=local;++i)row[i]=g.row[start+i]-first;
        std::vector<int>().swap(g.row);std::vector<int>().swap(g.col);std::vector<double>().swap(g.weight);std::vector<int>().swap(g.degree);
        std::vector<double> q(local),z(local),global(g.n);
        for(int i=0;i<local;++i)q[i]=(splitmix(uint64_t(seed)+start+i)>>11)*0x1.0p-53-0.5;
        double norm=std::sqrt(w.dot(q,q));if(norm==0)throw std::runtime_error("zero initial vector");for(double& x:q)x/=norm;
        struct History {int step;double lambda,residual,time;};std::vector<History> records;
        w.dot_compute=w.vector_compute=w.spmv_compute=w.allreduce=w.allgatherv=0;
        w.barrier();double begin=World::now(),lambda=0,residual=0;int completed=0;std::string reason="step_limit";
        for(int it=0;it<requested;++it){
            w.gather(q,global,count,disp);t=World::now();
#ifdef LANCZOS_MPI
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0;i<local;++i){double s=0;for(int j=row[i];j<row[i+1];++j)s+=weight[j]*global[col[j]];z[i]=kind=="lazy"?0.5*(s+global[start+i]):s;}
            w.spmv_compute+=World::now()-t;
            lambda=w.dot(q,z);double norm2=w.dot(z,z);double local_res=0;t=World::now();
#ifdef LANCZOS_MPI
            #pragma omp parallel for reduction(+:local_res) schedule(static)
#endif
            for(int i=0;i<local;++i){double d=z[i]-lambda*q[i];local_res+=d*d;}
            w.dot_compute+=World::now()-t;std::vector<double> sum(1,local_res);w.sum(sum);residual=std::sqrt(std::max(0.0,sum[0]));completed=it+1;
            if(!history.empty())records.push_back({completed,lambda,residual,World::now()-begin});
            if(tol>0&&residual<tol){reason="residual_tolerance";break;}
            if(norm2<=1e-30){reason="zero_operator_on_vector";break;}
            if(completed==requested)break;
            t=World::now();double inv=1/std::sqrt(norm2);
#ifdef LANCZOS_MPI
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0;i<local;++i)q[i]=z[i]*inv;
            w.vector_compute+=World::now()-t;
        }
        double kernel=w.maximum(World::now()-begin);
        double phases[5]={w.maximum(w.spmv_compute),w.maximum(w.vector_compute),w.maximum(w.dot_compute),w.maximum(w.allreduce),w.maximum(w.allgatherv)};
        w.gather(q,global,count,disp);double solve=w.maximum(World::now()-begin),e2e=w.maximum(World::now()-app),load_max=w.maximum(load);
        if(w.rank==0){std::cout<<std::setprecision(17)<<"backend: "<<(w.size>1?"mpi":"cpu")<<"\nN: "<<g.n<<"\ninput_E: "<<g.input_e
          <<"\noperator: "<<kind<<"\nseed: "<<seed<<"\nranks: "<<w.size<<"\nthreads: "<<w.threads<<"\nrequested_steps: "<<requested<<"\nsteps: "<<completed
          <<"\nstop_reason: "<<reason<<"\nlambda_max_ritz: "<<lambda<<"\nritz_residual: "<<residual<<"\niteration_s: "<<kernel<<"\nsolve_s: "<<solve
          <<"\nspmv_compute_s: "<<phases[0]<<"\nvector_compute_s: "<<phases[1]<<"\ndot_compute_s: "<<phases[2]
          <<"\nallreduce_s: "<<phases[3]<<"\nallgatherv_s: "<<phases[4]<<"\nload_s: "<<load_max<<"\nend_to_end_s: "<<e2e<<'\n';
          if(!dump.empty()){std::ofstream f(dump+".vector.txt");if(!f)throw std::runtime_error("cannot write vector");f<<std::setprecision(17);for(int i=0;i<g.n;++i)f<<i<<' '<<global[i]<<'\n';}
          if(!history.empty()){std::ofstream f(history);if(!f)throw std::runtime_error("cannot write history");f<<"step,rayleigh,direct_residual,local_elapsed_s\n"<<std::setprecision(17);for(auto h:records)f<<h.step<<','<<h.lambda<<','<<h.residual<<','<<h.time<<'\n';}
        }
#ifdef LANCZOS_MPI
        MPI_Finalize();
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';
#ifdef LANCZOS_MPI
        MPI_Abort(MPI_COMM_WORLD,1);
#endif
        return 1;
    }
}
