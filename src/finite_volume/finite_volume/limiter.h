#ifndef LIMITER_H
#define LIMITER_H

#include <gas/flow_state.h>
#include <grid/grid.h>
#include <util/vector3.h>
#include <util/types.h>

#include <memory>
#include <nlohmann/json.hpp>

using json = nlohmann::json;



template <typename T>
struct LimiterValues {
    LimiterValues() {}

    LimiterValues(int num_cells) {
        p = Field<T>("Limiter::p", num_cells);
        rho = Field<T>("Limiter::rho", num_cells);
        vx = Field<T>("Limiter::vx", num_cells);
        vy = Field<T>("Limiter::vy", num_cells);
        vz = Field<T>("Limiter::vz", num_cells);
    }

    Field<T> p;
    Field<T> rho;
    Field<T> vx;
    Field<T> vy;
    Field<T> vz;
};


template <typename T>
class Limiter {
public:
    virtual ~Limiter() {}

    Limiter() {}

    Limiter(bool enabled) : enabled_(enabled) {}

    virtual void calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits, const Cells<T>& cells,
                            const Interfaces<T>& faces, Vector3s<T>& grad);

    KOKKOS_INLINE_FUNCTION
    bool enabled() const { return enabled_; }

private:
    bool enabled_;
};


template <typename T>
class Unlimited : public Limiter<T> {
public:
    Unlimited() : Limiter<T>(true) {}

    ~Unlimited() {}

    void calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits, const Cells<T>& cells,
                            const Interfaces<T>& faces, Vector3s<T>& grad);
};


template <typename T>
class BarthJespersen : public Limiter<T> {
public:
    ~BarthJespersen() {}
    
    BarthJespersen() : Limiter<T>(true) {}

    BarthJespersen(double epsilon) : Limiter<T>(true), epsilon_(epsilon) {}

    void calculate_limiters(const Ibis::SubArray2D<T> values, Field<T>& limits, const Cells<T>& cells,
                            const Interfaces<T>& faces, Vector3s<T>& grid);


private:
    double epsilon_;
};

template<typename T>
std::unique_ptr<Limiter<T>> make_limiter(json config);

#endif
