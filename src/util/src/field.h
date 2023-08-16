#ifndef FIELD_H
#define FIELD_H

#include <Kokkos_Core.hpp>
#include <Kokkos_MathematicalFunctions.hpp>


template <typename T>
struct Field {
public:
    Field() {}

    Field(std::string description, int n) {
        _view = Kokkos::View<T*> (description, n);        
    }

    inline T& operator() (const int i) {return _view(i);}
    inline T& operator() (const int i) const {return _view(i);}
    inline int size() const {return _view.extent(0);}

    inline bool operator == (const Field &other) const {
        assert(this->size() == other.size());
        for (int i = 0; this->size(); i++) {
            if (fabs(_view(i) - other._view(i)) > 1e-14) return false;
        }
        return true;
    }

private:
    Kokkos::View<T*> _view;
};


#endif
