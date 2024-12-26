#include <vector>
#include <Kokkos_Core.hpp>
#include <parallel/mpi/ibis_mpi.h>
#include <util/numeric_types.h>

#ifdef Ibis_ENABLE_MPI

template <typename Scalar>
Scalar Ibis::Distributed::Min<Scalar>::reduce(Scalar& local_value) {
    Scalar global_min;
    auto mpi_type = Ibis::Distributed::MpiDataType<Scalar>::value();
    MPI_Allreduce(&local_value, &global_min, 1, mpi_type, MPI_MIN, comm_);
    return global_min;
}

// template <typename T>
// T Ibis::Distributed::Min<T>::execute(T* local_values, size_t num_values) {
//     std::vector<T> global_mins (num_values);
//     using mpi_type = Ibis::Distributed::MpiDataType<T>::type;
//     MPI_Allreduce(local_values, global_mins.data(), num_values, mpi_type, MPI_MIN, comm_);
//     return global_min;
// }
template struct Ibis::Distributed::Min<double>;

#endif
