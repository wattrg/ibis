#ifndef FIELD_H
#define FIELD_H

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>


template <typename T>
class Field {
public:
    Field() {}

    Field(std::string description, int n) {
        view_ = Kokkos::View<T*> (description, n);        
    }

    KOKKOS_FORCEINLINE_FUNCTION
    T& operator() (const int i) {return view_(i);}

    KOKKOS_FORCEINLINE_FUNCTION
    T& operator() (const int i) const {return view_(i);}

    KOKKOS_FORCEINLINE_FUNCTION
    int size() const {return view_.extent(0);}

    KOKKOS_FORCEINLINE_FUNCTION
    bool operator == (const Field &other) const {
        assert(this->size() == other.size());
        for (int i = 0; this->size(); i++) {
            if (Kokkos::fabs(view_(i) - other.view_(i)) > 1e-14) return false;
        }
        return true;
    }

private:
    Kokkos::View<T*> view_;
};


#endif
