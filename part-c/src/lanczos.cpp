#include "symmetric.hpp"
int main(int argc,char** argv){
    World w;
#ifdef LANCZOS_MPI
    int provided=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&w.rank);MPI_Comm_size(MPI_COMM_WORLD,&w.size);
    if(provided<MPI_THREAD_FUNNELED){MPI_Abort(MPI_COMM_WORLD,1);return 1;}
    omp_set_dynamic(0);
    #pragma omp parallel
    {
        #pragma omp single
        w.threads=omp_get_num_threads();
    }
    int tmin=0,tmax=0;
    MPI_Allreduce(&w.threads,&tmin,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
    MPI_Allreduce(&w.threads,&tmax,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
    if(tmin!=tmax){MPI_Abort(MPI_COMM_WORLD,1);return 1;}
#endif
    const double app_start=World::now();
    try{
        if(argc<3)throw std::runtime_error("Usage: lanczos[_mpi] GRAPH STEPS [--operator normalized|lazy] [--seed INT] [--reorth full|none] [--orthogonality 0|1] [--dump PREFIX] [--history CSV]");
        const std::string graph=argv[1];int requested=pr::positive(argv[2]),seed=20261029;bool reorth=true,orth=false;
        std::string dump,history,operator_kind="normalized";
        for(int a=3;a<argc;){std::string key=argv[a++];if(a==argc)throw std::runtime_error("missing option value");std::string value=argv[a++];
            if(key=="--seed"){auto s=pr::integer(value);if(s<0||s>INT_MAX)throw std::runtime_error("seed out of range");seed=int(s);}
            else if(key=="--reorth"){if(value!="full"&&value!="none")throw std::runtime_error("reorth: full or none");reorth=value=="full";}
            else if(key=="--orthogonality"){if(value!="0"&&value!="1")throw std::runtime_error("orthogonality: 0 or 1");orth=value=="1";}
            else if(key=="--operator"){if(value!="normalized"&&value!="lazy")throw std::runtime_error("operator: normalized or lazy");operator_kind=value;}
            else if(key=="--dump")dump=value;else if(key=="--history")history=value;
            else throw std::runtime_error("unknown option: "+key);
        }
        double t=World::now();Symmetric g(graph);double load=World::now()-t;
        int steps=std::min(requested,g.n);
        if(steps>1000)throw std::runtime_error("prototype limits basis to <=1000 steps; use 20-50 for experiments");
        std::vector<int> count(w.size),disp(w.size);
        for(int r=0;r<w.size;++r){disp[r]=int(int64_t(r)*g.n/w.size);count[r]=int(int64_t(r+1)*g.n/w.size)-disp[r];}
        int start=disp[w.rank],nlocal=count[w.rank];
        int first=g.row[start],last=g.row[start+nlocal],local_edges=last-first;
        std::vector<int> row(nlocal+1),col(g.col.begin()+first,g.col.begin()+last);
        std::vector<double> weight(g.weight.begin()+first,g.weight.begin()+last);
        for(int i=0;i<=nlocal;++i)row[i]=g.row[start+i]-first;
        std::vector<int>().swap(g.row);std::vector<int>().swap(g.col);std::vector<double>().swap(g.weight);std::vector<int>().swap(g.degree);
        auto spmv=[&](const std::vector<double>& q,std::vector<double>& out){
            double begin=World::now();
#ifdef LANCZOS_MPI
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0;i<nlocal;++i){double s=0;for(int j=row[i];j<row[i+1];++j)s+=weight[j]*q[col[j]];out[i]=operator_kind=="lazy"?0.5*(s+q[start+i]):s;}
            w.spmv_compute+=World::now()-begin;
        };
        std::vector<double> q(nlocal),global(g.n),z(nlocal),previous(nlocal,0);
        for(int i=0;i<nlocal;++i)q[i]=(splitmix(uint64_t(seed)+start+i)>>11)*0x1.0p-53-0.5;
        double norm=std::sqrt(w.dot(q,q));if(norm==0)throw std::runtime_error("zero starting vector");
        for(double& x:q)x/=norm;
        std::vector<std::vector<double>> basis;basis.reserve(steps);
        std::vector<double> alpha,beta;alpha.reserve(steps);beta.reserve(steps);
        double prev_beta=0;std::string reason="step_limit";
        // Initialization is excluded. Only iterative recurrence is in iteration_s.
        w.dot_compute=w.vector_compute=w.spmv_compute=w.allreduce=w.allgatherv=0;
        w.barrier();double begin=World::now();
        for(int j=0;j<steps;++j){
            basis.push_back(q);w.gather(q,global,count,disp);spmv(global,z);
            t=World::now();
#ifdef LANCZOS_MPI
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0;i<nlocal;++i)z[i]-=prev_beta*previous[i];
            w.vector_compute+=World::now()-t;
            double a=w.dot(q,z);alpha.push_back(a);t=World::now();
#ifdef LANCZOS_MPI
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0;i<nlocal;++i)z[i]-=a*q[i];
            w.vector_compute+=World::now()-t;
            if(reorth)for(int pass=0;pass<2;++pass){
                std::vector<double> coeff(j+1,0);t=World::now();
#ifdef LANCZOS_MPI
                #pragma omp parallel for schedule(static)
#endif
                for(int k=0;k<=j;++k){double s=0;for(int i=0;i<nlocal;++i)s+=basis[k][i]*z[i];coeff[k]=s;}
                w.dot_compute+=World::now()-t;w.sum(coeff);t=World::now();
#ifdef LANCZOS_MPI
                #pragma omp parallel for schedule(static)
#endif
                for(int i=0;i<nlocal;++i){double s=0;for(int k=0;k<=j;++k)s+=coeff[k]*basis[k][i];z[i]-=s;}
                w.vector_compute+=World::now()-t;
            }
            double b=std::sqrt(std::max(0.0,w.dot(z,z)));beta.push_back(b);
            if(b<=1e-13){reason="invariant_subspace_breakdown";break;}
            if(j+1==steps)break;
            t=World::now();previous=q;
#ifdef LANCZOS_MPI
            #pragma omp parallel for schedule(static)
#endif
            for(int i=0;i<nlocal;++i)q[i]=z[i]/b;
            w.vector_compute+=World::now()-t;prev_beta=b;
        }
        double kernel=w.maximum(World::now()-begin);
        // Snapshot phase maxima BEFORE numerical diagnostics to keep the timing contract.
        double phase_max[5]={w.maximum(w.spmv_compute),w.maximum(w.vector_compute),w.maximum(w.dot_compute),w.maximum(w.allreduce),w.maximum(w.allgatherv)};
        int m=int(alpha.size());t=World::now();Eigen eig=projected(alpha,beta,m);
        int best=int(std::max_element(eig.values.begin(),eig.values.end())-eig.values.begin());
        double lambda=eig.values[best];std::vector<double> v(nlocal,0);
        for(int k=0;k<m;++k)for(int i=0;i<nlocal;++i)v[i]+=basis[k][i]*eig.vectors[size_t(k)*m+best];
        double vn=std::sqrt(w.dot(v,v));if(vn==0)throw std::runtime_error("zero Ritz vector");for(double& x:v)x/=vn;
        w.gather(v,global,count,disp);spmv(global,z);
        for(int i=0;i<nlocal;++i)z[i]-=lambda*v[i];
        double residual=std::sqrt(w.dot(z,z));double orth_error=-1;
        if(orth){std::vector<double> gram(size_t(m)*m,0);
            for(int a=0;a<m;++a)for(int b=0;b<=a;++b){double s=0;for(int i=0;i<nlocal;++i)s+=basis[a][i]*basis[b][i];gram[size_t(a)*m+b]=s;}
            w.sum(gram);orth_error=0;for(int a=0;a<m;++a)for(int b=0;b<=a;++b)orth_error=std::max(orth_error,std::abs(gram[size_t(a)*m+b]-(a==b?1.0:0.0)));}
        double diagnostic=w.maximum(World::now()-t);double solve=w.maximum(World::now()-begin);double load_max=w.maximum(load),e2e=w.maximum(World::now()-app_start);
        if(w.rank==0){
            std::cout<<std::setprecision(17)<<"N: "<<g.n<<"\ninput_E: "<<g.input_e<<"\nundirected_edges: "<<g.undirected_e
              <<"\nranks: "<<w.size<<"\nthreads: "<<w.threads<<"\nrequested_steps: "<<requested<<"\nsteps: "<<m<<"\nseed: "<<seed
              <<"\noperator: "<<operator_kind<<"\nsolve_s: "<<solve<<"\nreorth: "<<(reorth?"full":"none")<<"\nstop_reason: "<<reason<<"\niteration_s: "<<kernel
              <<"\nspmv_compute_s: "<<phase_max[0]<<"\nvector_compute_s: "<<phase_max[1]<<"\ndot_compute_s: "<<phase_max[2]
              <<"\nallreduce_s: "<<phase_max[3]<<"\nallgatherv_s: "<<phase_max[4]
              <<"\nload_s: "<<load_max<<"\ndiagnostic_s: "<<diagnostic<<"\nend_to_end_s: "<<e2e
              <<"\nlambda_max_ritz: "<<lambda<<"\nritz_residual: "<<residual
              <<"\nritz_residual_estimate: "<<beta.back()*std::abs(eig.vectors[size_t(m-1)*m+best])
              <<"\northogonality_max: "<<orth_error<<"\n";
            if(!dump.empty()){
                std::ofstream f(dump+".vector.txt");if(!f)throw std::runtime_error("cannot write Ritz vector");f<<std::setprecision(17);
                for(int i=0;i<g.n;++i)f<<i<<' '<<global[i]<<'\n';
                std::ofstream tri(dump+".tridiagonal.csv");if(!tri)throw std::runtime_error("cannot write projection");
                tri<<"index,alpha,beta_next\n"<<std::setprecision(17);for(int i=0;i<m;++i)tri<<i<<','<<alpha[i]<<','<<beta[i]<<'\n';
            }
            if(!history.empty()){
                std::ofstream f(history);if(!f)throw std::runtime_error("cannot write history");f<<"step,largest_ritz,residual_estimate\n"<<std::setprecision(17);
                for(int k=1;k<=m;++k){auto e=projected(alpha,beta,k);int b=int(std::max_element(e.values.begin(),e.values.end())-e.values.begin());
                    f<<k<<','<<e.values[b]<<','<<beta[k-1]*std::abs(e.vectors[size_t(k-1)*k+b])<<'\n';}
            }
        }
        (void)local_edges;
#ifdef LANCZOS_MPI
        MPI_Finalize();
#endif
        return 0;
    }catch(const std::exception& e){std::cerr<<"rank "<<w.rank<<": "<<e.what()<<'\n';
#ifdef LANCZOS_MPI
        MPI_Abort(MPI_COMM_WORLD,1);
#endif
        return 1;
    }
}
