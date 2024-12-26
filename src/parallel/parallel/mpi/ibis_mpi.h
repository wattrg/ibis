#ifndef IBIS_MPI_H
#define IBIS_MPI_H

#ifdef Ibis_ENABLE_MPI

#include <mpi.h>

namespace Ibis{
namespace Distributed{

// Used as template parameter to determine which distributed memory
// paradigm to use
struct Mpi;

// MPI data types
template <typename Type>
struct MpiDataType;

template <typename Type>
struct MpiDataType;

#define MpiTypeMapping(type, MPI_type)                \
template <>                                           \
struct MpiDataType<type> {                            \
    static MPI_Datatype value() { return MPI_type; }  \
};                                                    \

MpiTypeMapping(short int, MPI_SHORT)
MpiTypeMapping(int, MPI_INT)
MpiTypeMapping(long int, MPI_LONG)
MpiTypeMapping(long long int, MPI_LONG_LONG)
MpiTypeMapping(unsigned char, MPI_UNSIGNED_CHAR)
MpiTypeMapping(unsigned short int, MPI_UNSIGNED_SHORT)
MpiTypeMapping(unsigned int, MPI_UNSIGNED)
MpiTypeMapping(unsigned long int, MPI_UNSIGNED_LONG)
MpiTypeMapping(unsigned long long int, MPI_UNSIGNED_LONG_LONG)
MpiTypeMapping(float, MPI_FLOAT)
MpiTypeMapping(double, MPI_DOUBLE)
MpiTypeMapping(long double, MPI_LONG_DOUBLE)
MpiTypeMapping(char, MPI_CHAR)


// Reductions
template <typename Scalar>
struct Min {
public:
    Min() : comm_(MPI_COMM_WORLD) {}

    Min(MPI_Comm comm) : comm_(comm) {}

    Scalar reduce(Scalar& local_value);

    // Scalar reduce(Scalar* local_values, size_t num_values);

public:
    using scalar_type = Scalar;
    using shared_reducer = Kokkos::Min<Scalar>;

private:
    MPI_Comm comm_;
};


 
}
}

#endif
#endif
