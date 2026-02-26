#ifndef IBIS_MPI_TYPES_H
#define IBIS_MPI_TYPES_H

#include <mpi.h>

namespace Ibis {

// MPI data types
template <typename Type>
struct MpiDataType;

template <typename Type>
struct MpiDataType;

#define MpiTypeMapping(type, MPI_type)                   \
    template <>                                          \
    struct MpiDataType<type> {                           \
        static MPI_Datatype value() { return MPI_type; } \
    };

MpiTypeMapping(short int, MPI_SHORT)                                // NOLINT
    MpiTypeMapping(int, MPI_INT)                                    // NOLINT
    MpiTypeMapping(long int, MPI_LONG)                              // NOLINT
    MpiTypeMapping(long long int, MPI_LONG_LONG)                    // NOLINT
    MpiTypeMapping(unsigned char, MPI_UNSIGNED_CHAR)                // NOLINT
    MpiTypeMapping(unsigned short int, MPI_UNSIGNED_SHORT)          // NOLINT
    MpiTypeMapping(unsigned int, MPI_UNSIGNED)                      // NOLINT
    MpiTypeMapping(unsigned long int, MPI_UNSIGNED_LONG)            // NOLINT
    MpiTypeMapping(unsigned long long int, MPI_UNSIGNED_LONG_LONG)  // NOLINT
    MpiTypeMapping(float, MPI_FLOAT)                                // NOLINT
    MpiTypeMapping(double, MPI_DOUBLE)                              // NOLINT
    MpiTypeMapping(long double, MPI_LONG_DOUBLE)                    // NOLINT
    MpiTypeMapping(char, MPI_CHAR)                                  // NOLINT

    // Custom MPI operations of standard overloaded operators
    template <typename T>
    void MPI_custom_sum(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
    (void)datatype;
    for (int i = 0; i < *len; i++) {
        inoutvec[i] += invec[i];
    }
}

template <typename T>
void MPI_custom_max(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
    (void)datatype;
    for (int i = 0; i < *len; i++) {
        inoutvec[i] = max(invec[i], inoutvec[i]);
    }
}

template <typename T>
void MPI_custom_min(T* invec, T* inoutvec, int* len, MPI_Datatype* datatype) {
    (void)datatype;
    for (int i = 0; i < *len; i++) {
        inoutvec[i] = min(invec[i], inoutvec[i]);
    }
}

}  // namespace Ibis

#endif
