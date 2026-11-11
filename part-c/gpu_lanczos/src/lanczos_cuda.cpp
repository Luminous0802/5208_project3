#include "../../src/symmetric.hpp"
#include "device_lanczos.hpp"
#include <cstdlib>
#include <cstring>

int main(int argc,char** argv){
    World w;int provided=0;
    const char* rank_env=std::getenv("PMI_RANK");
    std::cerr<<"GPU_LANCZOS_ENTER_MPI rank="<<(rank_env?rank_env:"?")<<'\n';
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    MPI_Comm_rank(MPI_COMM_WORLD,&w.rank);MPI_Comm_size(MPI_COMM_WORLD,&w.size);
    if(provided<MPI_THREAD_FUNNELED){MPI_Abort(MPI_COMM_WORLD,1);return 1;}
    omp_set_dynamic(0);
    #pragma omp parallel
    {
        #pragma omp single
        w.threads=omp_get_num_threads();
    }
    try{
        int tmin=0,tmax=0;
        MPI_Allreduce(&w.threads,&tmin,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        MPI_Allreduce(&w.threads,&tmax,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if(tmin!=tmax)throw std::runtime_error("Inconsistent OpenMP thread count");
        char version[MPI_MAX_LIBRARY_VERSION_STRING];int length=0;MPI_Get_library_version(version,&length);
        std::string library(version,length);
        if(library.find("MPICH")==std::string::npos)throw std::runtime_error("This launcher requires a fresh MPICH build");
        if(w.rank==0)std::cerr<<"MPI_LIBRARY "<<library<<'\n';
        const double app_start=World::now();
        if(argc<3)throw std::runtime_error("Usage: lanczos_cuda_mpi GRAPH STEPS [--operator lazy] [--reorth full] [--seed INT] [--block-size 256] [--dump PREFIX] [--orthogonality 0|1]");
        std::string graph=argv[1],dump;int requested=pr::positive(argv[2]),seed=20261029,block=256;bool orth=false;
        for(int i=3;i<argc;){std::string key=argv[i++];if(i==argc)throw std::runtime_error("Missing option value");std::string value=argv[i++];
            if(key=="--seed"){auto n=pr::integer(value);if(n<0||n>INT_MAX)throw std::runtime_error("Seed out of range");seed=int(n);}
            else if(key=="--operator"){if(value!="lazy")throw std::runtime_error("Only the common lazy operator is supported");}
            else if(key=="--reorth"){if(value!="full")throw std::runtime_error("GPU comparison requires two-pass full reorthogonalization");}
            else if(key=="--block-size")block=pr::positive(value);
            else if(key=="--dump")dump=value;
            else if(key=="--orthogonality"){if(value!="0"&&value!="1")throw std::runtime_error("Orthogonality must be 0 or 1");orth=value=="1";}
            else throw std::runtime_error("Unknown option: "+key);
        }
        double t=World::now();Symmetric g(graph);double load_s=World::now()-t;
        int steps=std::min(requested,g.n);if(steps>1000)throw std::runtime_error("Basis limited to 1000 steps");
        std::vector<int> count(w.size),disp(w.size);
        for(int r=0;r<w.size;++r){disp[r]=int(int64_t(r)*g.n/w.size);count[r]=int(int64_t(r+1)*g.n/w.size)-disp[r];}
        int start=disp[w.rank],nlocal=count[w.rank],first=g.row[start],last=g.row[start+nlocal];
        std::vector<int> row(nlocal+1),col(g.col.begin()+first,g.col.begin()+last);
        std::vector<double> weight(g.weight.begin()+first,g.weight.begin()+last);
        for(int i=0;i<=nlocal;++i)row[i]=g.row[start+i]-first;
        std::vector<int>().swap(g.row);std::vector<int>().swap(g.col);std::vector<int>().swap(g.degree);std::vector<double>().swap(g.weight);
        std::vector<double> q(nlocal),global(g.n);
        for(int i=0;i<nlocal;++i)q[i]=(splitmix(uint64_t(seed)+start+i)>>11)*0x1.0p-53-0.5;
        double initial_norm=std::sqrt(w.dot(q,q));if(!std::isfinite(initial_norm)||initial_norm==0)throw std::runtime_error("Invalid initial vector");
        for(double& x:q)x/=initial_norm;
        t=World::now();DeviceLanczos device(g.n,start,nlocal,steps,block,row,col,weight);double setup_s=World::now()-t;
        const GpuInfo info=device.info();
        std::cerr<<"CUDA_READY rank="<<w.rank<<" local_N="<<nlocal<<" device="<<info.name<<" uuid="<<info.uuid<<'\n';
        // Require every rank to share the same physical GPU on the same host.
        char identity[MPI_MAX_PROCESSOR_NAME+64]={};int host_length=0;MPI_Get_processor_name(identity,&host_length);
        std::string tag=std::string(identity,host_length)+"/"+info.uuid;
        std::fill(identity,identity+sizeof(identity),0);if(tag.size()>=sizeof(identity))throw std::runtime_error("Device identity too long");
        std::copy(tag.begin(),tag.end(),identity);
        std::vector<char> identities(std::size_t(w.size)*sizeof(identity));
        MPI_Allgather(identity,int(sizeof(identity)),MPI_CHAR,identities.data(),int(sizeof(identity)),MPI_CHAR,MPI_COMM_WORLD);
        for(int r=0;r<w.size;++r)if(std::string(identities.data()+std::size_t(r)*sizeof(identity))!=tag)throw std::runtime_error("This experiment requires ranks on one host sharing exactly one GPU");
        device.initialize(q);w.gather(q,global,count,disp);
        // Exercise every kernel before timing; discard this state, restore q/previous.
        t=World::now();device.upload_global(global);device.store_basis(0);device.spmv();device.subtract_previous(0);
        device.subtract_q(device.dot_qz());
        for(int pass=0;pass<2;++pass)device.subtract_basis(device.basis_dots(1));
        (void)device.dot_zz();device.advance(1);device.reconstruct(std::vector<double>{1});(void)device.dot_qq();device.scale_q(1);
        device.initialize(q);device.synchronize();double warmup_s=World::now()-t;
        const double pre_h2d=device.transfers().h2d;device.reset_measurement();
        auto sum=[&](double local){std::vector<double> v{local};w.sum(v);if(!std::isfinite(v[0]))throw std::runtime_error("Nonfinite MPI-reduced scalar");return v[0];};
        std::vector<double> alpha,beta;alpha.reserve(steps);beta.reserve(steps);double previous_beta=0;std::string reason="step_limit";
        w.allreduce=w.allgatherv=0;w.barrier();double begin=World::now();
        for(int j=0;j<steps;++j){
            device.store_basis(j);device.download_q(q);w.gather(q,global,count,disp);device.upload_global(global);device.spmv();
            device.subtract_previous(previous_beta);double a=sum(device.dot_qz());alpha.push_back(a);device.subtract_q(a);
            for(int pass=0;pass<2;++pass){auto coeff=device.basis_dots(j+1);w.sum(coeff);if(!std::all_of(coeff.begin(),coeff.end(),[](double x){return std::isfinite(x);}))throw std::runtime_error("Nonfinite reorthogonalization coefficient");device.subtract_basis(coeff);}
            double b=std::sqrt(std::max(0.0,sum(device.dot_zz())));beta.push_back(b);
            if(b<=1e-13){reason="invariant_subspace_breakdown";break;}
            if(j+1==steps)break;
            device.advance(b);previous_beta=b;
        }
        device.synchronize();double iteration_s=w.maximum(World::now()-begin);
        std::size_t iteration_events=device.event_count();auto iteration_transfers=device.transfers();
        double iteration_reduce=w.maximum(w.allreduce),iteration_gather=w.maximum(w.allgatherv);
        double diagnostic_start=World::now();int m=int(alpha.size());Eigen eig=projected(alpha,beta,m);
        int best=int(std::max_element(eig.values.begin(),eig.values.end())-eig.values.begin());double lambda=eig.values[best];
        std::vector<double> coeff(m);for(int k=0;k<m;++k)coeff[k]=eig.vectors[std::size_t(k)*m+best];
        device.reconstruct(coeff);double norm=std::sqrt(std::max(0.0,sum(device.dot_qq())));device.scale_q(norm);
        device.download_q(q);w.gather(q,global,count,disp);device.upload_global(global);device.spmv();device.subtract_q(lambda);
        double residual=std::sqrt(std::max(0.0,sum(device.dot_zz()))),orth_error=-1;
        if(orth){orth_error=0;for(int j=0;j<m;++j){auto gram=device.gram_column(j,m);w.sum(gram);for(int k=0;k<m;++k){if(!std::isfinite(gram[k]))throw std::runtime_error("Nonfinite Gram matrix");orth_error=std::max(orth_error,std::abs(gram[k]-(k==j?1.0:0.0)));}}}
        device.synchronize();double diagnostic_s=w.maximum(World::now()-diagnostic_start),solve_s=w.maximum(World::now()-begin);
        double e2e=w.maximum(World::now()-app_start);auto solve_transfers=device.transfers();
        // Read timing events after stopping the solve timer. These are event
        // intervals, not claims about exclusive GPU utilization or overlap.
        double iteration_gpu_events=w.maximum(device.event_seconds(iteration_events));
        double solve_gpu_events=w.maximum(device.event_seconds(device.event_count()));
        double h2d=w.maximum(solve_transfers.h2d),d2h=w.maximum(solve_transfers.d2h);
        double iter_h2d=w.maximum(iteration_transfers.h2d),iter_d2h=w.maximum(iteration_transfers.d2h);
        setup_s=w.maximum(setup_s);warmup_s=w.maximum(warmup_s);load_s=w.maximum(load_s);
        double setup_h2d=w.maximum(pre_h2d);
        if(!std::isfinite(lambda)||!std::isfinite(residual)||device.event_count()==0)throw std::runtime_error("Invalid GPU result");
        if(w.rank==0){
            std::cout<<std::setprecision(17)<<"backend: cuda_lanczos\nprecision: fp64\nexecution_mode: MPI+CUDA\nmpi_implementation: MPICH\n"
                <<"N: "<<g.n<<"\ninput_E: "<<g.input_e<<"\nundirected_edges: "<<g.undirected_e<<"\nranks: "<<w.size<<"\nthreads: "<<w.threads
                <<"\noperator: lazy\nreorth: full\nseed: "<<seed<<"\nrequested_steps: "<<requested<<"\nsteps: "<<m<<"\nstop_reason: "<<reason
                <<"\niteration_s: "<<iteration_s<<"\nsolve_s: "<<solve_s<<"\nend_to_end_s: "<<e2e<<"\ndiagnostic_s: "<<diagnostic_s<<"\nload_s: "<<load_s
                <<"\ngpu_setup_s: "<<setup_s<<"\nwarmup_s: "<<warmup_s<<"\nsetup_h2d_s: "<<setup_h2d
                <<"\niteration_h2d_s: "<<iter_h2d<<"\niteration_d2h_s: "<<iter_d2h<<"\nsolve_h2d_s: "<<h2d<<"\nsolve_d2h_s: "<<d2h
                <<"\niteration_gpu_event_s: "<<iteration_gpu_events<<"\nsolve_gpu_event_s: "<<solve_gpu_events
                <<"\niteration_allreduce_s: "<<iteration_reduce<<"\niteration_allgatherv_s: "<<iteration_gather
                <<"\ngpu_kernel_calls: "<<device.event_count()<<"\ngpu_name: "<<info.name<<"\ngpu_uuid: "<<info.uuid<<"\nshared_physical_gpus: 1"
                <<"\ncompute_capability: "<<info.capability<<"\ncuda_runtime_version: "<<info.runtime<<"\ncuda_driver_version: "<<info.driver
                <<"\ngpu_memory_bytes: "<<info.total_memory<<"\nblock_size: "<<block
                <<"\nlambda_max_ritz: "<<lambda<<"\nritz_residual: "<<residual<<"\northogonality_max: "<<orth_error<<'\n';
            if(!dump.empty()){
                std::ofstream f(dump+".vector.txt");if(!f)throw std::runtime_error("Cannot write vector");f<<std::setprecision(17);for(int i=0;i<g.n;++i)f<<i<<' '<<global[i]<<'\n';
                std::ofstream tri(dump+".tridiagonal.csv");if(!tri)throw std::runtime_error("Cannot write projection");tri<<"index,alpha,beta_next\n"<<std::setprecision(17);for(int j=0;j<m;++j)tri<<j<<','<<alpha[j]<<','<<beta[j]<<'\n';
            }
        }
        MPI_Finalize();return 0;
    }catch(const std::exception& e){std::cerr<<"GPU Lanczos rank "<<w.rank<<": "<<e.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;}
}
