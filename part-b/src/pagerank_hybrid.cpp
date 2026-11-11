#include "graph.hpp"
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>

struct Config {
    std::string graph, part="vertex", schedule="static", dump, profile;
    int k=50, chunk=0;
    double tol=0;
};
Config parse(int argc, char** argv) {
    if (argc < 3) throw std::runtime_error(
        "Usage: pagerank_hybrid GRAPH K [TOL] [--partition vertex|edge] "
        "[--schedule static|dynamic] [--chunk N] [--dump FILE] [--profile FILE]");
    Config c; c.graph=argv[1]; c.k=pr::positive(argv[2]); int a=3;
    if (a<argc && std::string(argv[a]).rfind("--",0)!=0) {
        size_t end=0; std::string s=argv[a++]; c.tol=std::stod(s,&end);
        if (end!=s.size() || !std::isfinite(c.tol) || c.tol<=0)
            throw std::runtime_error("TOL must be positive and finite");
    }
    while (a<argc) {
        std::string key=argv[a++];
        if (a==argc) throw std::runtime_error("missing option value: "+key);
        std::string val=argv[a++];
        if (key=="--partition") c.part=val;
        else if (key=="--schedule") c.schedule=val;
        else if (key=="--chunk") c.chunk=pr::positive(val);
        else if (key=="--dump") c.dump=val;
        else if (key=="--profile") c.profile=val;
        else throw std::runtime_error("unknown option: "+key);
    }
    if (c.part!="vertex" && c.part!="edge") throw std::runtime_error("partition: vertex or edge");
    if (c.schedule!="static" && c.schedule!="dynamic") throw std::runtime_error("schedule: static or dynamic");
    if (c.schedule=="dynamic" && c.chunk==0) c.chunk=64;
    return c;
}
const char* env(const char* name) { const char* s=std::getenv(name); return s?s:"unset"; }
int main(int argc, char** argv) {
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    int me=0,p=0; MPI_Comm_rank(MPI_COMM_WORLD,&me); MPI_Comm_size(MPI_COMM_WORLD,&p);
    const double application_start=MPI_Wtime();
    try {
        if (provided<MPI_THREAD_FUNNELED) throw std::runtime_error("MPI_THREAD_FUNNELED unavailable");
        Config c=parse(argc,argv);
        omp_set_dynamic(0);
        // static,0 means implementation's contiguous static schedule, with no explicit chunk.
        omp_set_schedule(c.schedule=="static"?omp_sched_static:omp_sched_dynamic,c.chunk);
        int threads=1;
        #pragma omp parallel
        {
            #pragma omp single
            threads=omp_get_num_threads();
        }
        int tmin=0,tmax=0;
        MPI_Allreduce(&threads,&tmin,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        MPI_Allreduce(&threads,&tmax,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if (tmin!=tmax) throw std::runtime_error("inconsistent thread counts between ranks");
        double phase=MPI_Wtime(); auto g=pr::load(c.graph);
        const double load_s=MPI_Wtime()-phase;
        phase=MPI_Wtime();
        auto b=pr::partition(g,p,c.part=="edge");
        std::vector<int> count(p), disp(p);
        for (int r=0;r<p;++r) { disp[r]=b[r]; count[r]=b[r+1]-b[r]; }
        const int start=b[me], nlocal=count[me], n=g.n, e=g.e;
        const int first=g.row[start], last=g.row[start+nlocal], elocal=last-first;
        std::vector<int> row(nlocal+1),col(g.col.begin()+first,g.col.begin()+last);
        for (int i=0;i<=nlocal;++i) row[i]=g.row[start+i]-first;
        std::vector<int>().swap(g.row); std::vector<int>().swap(g.col);
        const double partition_s=MPI_Wtime()-phase;
        std::vector<double> rank(n,1.0/n),next(n,0),local(nlocal);
        double compute=0,reduce=0,gather=0,residual=0; int completed=0;
        MPI_Barrier(MPI_COMM_WORLD);
        const double begin=MPI_Wtime();
        for (int it=0;it<c.k;++it) {
            phase=MPI_Wtime(); double mass_local=0;
            #pragma omp parallel for schedule(static) reduction(+:mass_local)
            for (int i=0;i<nlocal;++i) {
                int u=start+i;
                if (g.out[u]==0) mass_local+=rank[u];
            }
            compute+=MPI_Wtime()-phase;
            phase=MPI_Wtime(); double mass=0;
            MPI_Allreduce(&mass_local,&mass,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            reduce+=MPI_Wtime()-phase;
            phase=MPI_Wtime();
            #pragma omp parallel for schedule(runtime)
            for (int i=0;i<nlocal;++i) {
                double incoming=0;
                for (int j=row[i];j<row[i+1];++j) {
                    int u=col[j]; incoming+=rank[u]/g.out[u];
                }
                local[i]=(1.0-0.85)/n+0.85*(incoming+mass/n);
            }
            compute+=MPI_Wtime()-phase;
            phase=MPI_Wtime();
            MPI_Allgatherv(local.empty()?nullptr:local.data(),nlocal,MPI_DOUBLE,
                next.data(),count.data(),disp.data(),MPI_DOUBLE,MPI_COMM_WORLD);
            gather+=MPI_Wtime()-phase;
            phase=MPI_Wtime(); double diff=0;
            #pragma omp parallel for schedule(static) reduction(+:diff)
            for (int i=0;i<nlocal;++i) diff+=std::abs(next[start+i]-rank[start+i]);
            compute+=MPI_Wtime()-phase;
            phase=MPI_Wtime();
            // Preserve A's two-Allreduce protocol even for fixed-K runs.
            MPI_Allreduce(&diff,&residual,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            reduce+=MPI_Wtime()-phase;
            rank.swap(next); completed=it+1;
            if (c.tol>0 && residual<c.tol) break;
        }
        double elapsed=MPI_Wtime()-begin;
        // E2E includes post-init setup, loading and iterations; excludes final diagnostics/I/O.
        double end_to_end=MPI_Wtime()-application_start;
        double values[7]={elapsed,compute,reduce,gather,load_s,partition_s,end_to_end},maxima[7]={};
        MPI_Reduce(values,maxima,7,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
        int sizes[2]={nlocal,elocal},mins[2]={},maxs[2]={};
        MPI_Reduce(sizes,mins,2,MPI_INT,MPI_MIN,0,MPI_COMM_WORLD);
        MPI_Reduce(sizes,maxs,2,MPI_INT,MPI_MAX,0,MPI_COMM_WORLD);
        if (!c.profile.empty()) {
            double info[7]={double(start),double(nlocal),double(elocal),compute,reduce,gather,elapsed};
            std::vector<double> all(me==0?size_t(p)*7:0);
            MPI_Gather(info,7,MPI_DOUBLE,me==0?all.data():nullptr,7,MPI_DOUBLE,0,MPI_COMM_WORLD);
            char host[MPI_MAX_PROCESSOR_NAME]; int len=0; MPI_Get_processor_name(host,&len);
            std::vector<char> hosts(me==0?size_t(p)*MPI_MAX_PROCESSOR_NAME:0);
            // Zero padding avoids exporting uninitialized bytes.
            std::fill(host+len,host+MPI_MAX_PROCESSOR_NAME,0);
            MPI_Gather(host,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,me==0?hosts.data():nullptr,
                MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,MPI_COMM_WORLD);
            if (me==0) {
                std::ofstream out(c.profile); if (!out) throw std::runtime_error("cannot write profile");
                out<<"rank,hostname,threads,start,vertices,edges,compute_s,allreduce_s,allgatherv_s,iteration_s\n";
                out<<std::setprecision(17);
                for(int r=0;r<p;++r) {
                    out<<r<<','<<(hosts.data()+r*MPI_MAX_PROCESSOR_NAME)<<','<<threads;
                    for(int j=0;j<7;++j) out<<','<<all[r*7+j];
                    out<<'\n';
                }
            }
        }
        if(me==0) {
            double sum=std::accumulate(rank.begin(),rank.end(),0.0);
            std::cout<<std::setprecision(17);
            std::cout<<"N: "<<n<<"\nE: "<<e<<"\nranks: "<<p<<"\nthreads: "<<threads
                <<"\npartition: "<<c.part<<"\nschedule: "<<c.schedule<<"\nchunk: "<<c.chunk
                <<"\nmpi_thread_provided: "<<provided<<"\niterations: "<<completed
                <<"\niteration_s: "<<maxima[0]<<"\ncompute_s: "<<maxima[1]
                <<"\nallreduce_s: "<<maxima[2]<<"\nallgatherv_s: "<<maxima[3]
                <<"\nload_s: "<<maxima[4]<<"\npartition_s: "<<maxima[5]
                <<"\nend_to_end_s: "<<maxima[6]
                <<"\nlocal_vertices_min: "<<mins[0]<<"\nlocal_vertices_max: "<<maxs[0]
                <<"\nlocal_edges_min: "<<mins[1]<<"\nlocal_edges_max: "<<maxs[1]
                <<"\nedge_imbalance: "<<(e?double(maxs[1])*p/e:0)
                <<"\nrank_sum: "<<sum<<"\nresidual: "<<residual
                <<"\nOMP_PROC_BIND: "<<env("OMP_PROC_BIND")<<"\nOMP_PLACES: "<<env("OMP_PLACES")<<'\n';
            if (n<=20) for(int v=0;v<n;++v) std::cout<<"rank "<<v<<": "<<rank[v]<<'\n';
            if(!c.dump.empty()) {
                std::ofstream out(c.dump); if(!out) throw std::runtime_error("cannot write rank dump");
                out<<std::setprecision(17);
                for(int v=0;v<n;++v) out<<v<<' '<<rank[v]<<'\n';
            }
        }
        MPI_Finalize(); return 0;
    } catch(const std::exception& ex) {
        std::cerr<<"rank "<<me<<": "<<ex.what()<<'\n';
        MPI_Abort(MPI_COMM_WORLD,1); return 1;
    }
}
