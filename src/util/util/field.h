#ifndef FIELD_H
#define FIELD_H

#include <Kokkos_Core.hpp>
#include <vector>

template <typename T, class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
class Field {
public:
    using view_type = Kokkos::View<T*, Layout, Space>;
    using array_layout = typename view_type::array_layout;
    using memory_space = typename view_type::memory_space;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_space = typename mirror_view_type::memory_space;
    using mirror_type = Field<T, mirror_layout, mirror_space>;

public:
    Field() {}

    Field(std::string description, size_t n) { view_ = view_type(description, n); }

    Field(std::string description, std::vector<T> values) {
        view_ = view_type(description, values.size());
        auto view_host = Kokkos::create_mirror_view(view_);
        for (size_t i = 0; i < values.size(); i++) {
            view_host(i) = values[i];
        }
        Kokkos::deep_copy(view_, view_host);
    }

    Field(view_type values) : view_(values) {} 

    KOKKOS_FORCEINLINE_FUNCTION
    T& operator()(const size_t i) { return view_(i); }

    KOKKOS_FORCEINLINE_FUNCTION
    T& operator()(const size_t i) const { return view_(i); }

    KOKKOS_FORCEINLINE_FUNCTION
    size_t size() const { return view_.extent(0); }

    KOKKOS_FORCEINLINE_FUNCTION
    bool operator==(const Field& other) const {
        assert(this->size() == other.size());
        for (size_t i = 0; this->size(); i++) {
            if (Kokkos::fabs(view_(i) - other.view_(i)) > 1e-14) return false;
        }
        return true;
    }

    mirror_type host_mirror() const {
        auto mirror_view = Kokkos::create_mirror_view(view_);
        return mirror_type(mirror_view);
    }

    template <class OtherSpace>
    void deep_copy(const Field<T, Layout, OtherSpace>& other) {
        Kokkos::deep_copy(view_, other.view_);
    }

    void deep_copy(T value) { Kokkos::deep_copy(view_, value); }

    std::string to_string() const {
        std::string result = "Field(";
        for (size_t i = 0; i < view_.size(); i++) {
            result.append(std::to_string(view_(i)));
            result.append(",");
        }
        result.append(")");
        return result;
    }

public:
    view_type view_;
};

#endif
