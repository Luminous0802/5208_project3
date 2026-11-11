// Diagnostic only: exercise the SAME World reductions/gather as the solvers.
// This is not a performance measurement and does not replace solver validation.
#include "../src/symmetric.hpp"
#include <cstdlib>

static void stage(const char* message, int rank) {
    std::cerr << "MPI_PROBE rank=" << rank << " " << message << std::endl;
}

int main(int argc, char** argv) {
    const char* rank_env=std::getenv("OMPI_COMM_WORLD_RANK");
    if(!rank_env)rank_env=std::getenv("PMI_RANK");
    int before_rank=rank_env?std::atoi(rank_env):-1;
    stage("ENTER_MPI_INIT_THREAD",before_rank);
    int provided=0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    World w;
    MPI_Comm_rank(MPI_COMM_WORLD,&w.rank);
    MPI_Comm_size(MPI_COMM_WORLD,&w.size);
    stage("EXIT_MPI_INIT_THREAD",w.rank);
    if(provided<MPI_THREAD_FUNNELED) {
        std::cerr << "Insufficient MPI thread support: " << provided << std::endl;
        MPI_Abort(MPI_COMM_WORLD,1);
    }
    char version[MPI_MAX_LIBRARY_VERSION_STRING];int length=0;
    MPI_Get_library_version(version,&length);
    if(w.rank==0)std::cerr << "MPI_LIBRARY " << std::string(version,length) << std::endl;
    omp_set_dynamic(0);
    stage("ENTER_OPENMP",w.rank);
    #pragma omp parallel
    {
        #pragma omp single
        w.threads=omp_get_num_threads();
    }
    std::cerr << "MPI_PROBE rank=" << w.rank << " EXIT_OPENMP threads=" << w.threads << std::endl;
    if(w.threads!=2)MPI_Abort(MPI_COMM_WORLD,2);
    for(int n : {w.size*3,3}) {
        std::vector<int> count(w.size),disp(w.size);
        for(int r=0;r<w.size;++r) {
            disp[r]=int(int64_t(r)*n/w.size);
            count[r]=int(int64_t(r+1)*n/w.size)-disp[r];
        }
        std::vector<double> q(count[w.rank]),global(n);
        for(int i=0;i<count[w.rank];++i)q[i]=disp[w.rank]+i+1;
        std::cerr << "MPI_PROBE rank=" << w.rank << " N=" << n << " local=" << q.size() << std::endl;
        stage("ENTER_DOT_ALLREDUCE",w.rank);
        double norm2=w.dot(q,q);
        stage("EXIT_DOT_ALLREDUCE",w.rank);
        if(std::abs(norm2-n*(n+1.0)*(2*n+1)/6)>1e-10)MPI_Abort(MPI_COMM_WORLD,3);
        stage("ENTER_ALLGATHERV",w.rank);
        w.gather(q,global,count,disp);
        stage("EXIT_ALLGATHERV",w.rank);
        for(int i=0;i<n;++i)if(global[i]!=i+1)MPI_Abort(MPI_COMM_WORLD,4);
        stage("ENTER_MAX_ALLREDUCE",w.rank);
        if(w.maximum(w.rank)!=w.size-1)MPI_Abort(MPI_COMM_WORLD,5);
        stage("EXIT_MAX_ALLREDUCE",w.rank);
        stage("ENTER_BARRIER",w.rank);
        w.barrier();
        stage("EXIT_BARRIER",w.rank);
    }
    stage("ENTER_MPI_FINALIZE",w.rank);
    MPI_Finalize();
    stage("EXIT_MPI_FINALIZE",w.rank);
    if(w.rank==0)std::cout << "MPI_EIGEN_PROBE_PASS ranks=" << w.size << " threads=2\n";
    return 0;
}
