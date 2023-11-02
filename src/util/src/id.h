#ifndef ID_H
#define ID_H

#include "Kokkos_Core_fwd.hpp"
#include <vector>
#include <Kokkos_Core.hpp>

struct IdConstructor {
public:
    IdConstructor() {
        _ids = std::vector<int> {};
        _offsets = std::vector<int> {0};
    }

    void push_back(std::vector<int> ids) {
        for (unsigned int i = 0; i < ids.size(); i++) {
            _ids.push_back(ids[i]);
        }
        _offsets.push_back(_offsets.back() + ids.size());
    }

    std::vector<int> ids() const {return _ids;}
    std::vector<int> offsets() const {return _offsets;}

private:
    std::vector<int> _ids;
    std::vector<int> _offsets;
};


template <class Layout=Kokkos::DefaultExecutionSpace::array_layout,
          class Space=Kokkos::DefaultExecutionSpace::memory_space>
struct Id {
public:
    using view_type = Kokkos::View<int*, Layout, Space>;
    using array_layout = typename view_type::array_layout;
    using memory_space = typename view_type::memory_space;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_space = typename mirror_view_type::memory_space;
    using mirror_type = Id<mirror_layout, mirror_space>;

public:
    Id() {}

    Id(view_type ids, view_type offsets){
        _ids = view_type ("Id::id_from_views", static_cast<int>(ids.size()));
        _offsets = view_type ("Id::offset", static_cast<int>(offsets.size()));
        Kokkos::deep_copy(_ids, ids);
        Kokkos::deep_copy(_offsets, offsets);
    }

    Id(std::vector<int> ids, std::vector<int> offsets){
        _ids = view_type ("Id::id_from_vec", static_cast<int>(ids.size()));
        _offsets = view_type ("Id::offset", static_cast<int>(offsets.size()));

        // make of copy of the view's on the host so we can copy
        // from std::vector
        mirror_view_type ids_mirror ("Id::id::mirror", 
                                     static_cast<int>(ids.size()));
        mirror_view_type offsets_mirror ("Id::offset::mirror", 
                                         static_cast<int>(offsets.size()));
        for (unsigned int i = 0; i < ids.size(); i++) {
            ids_mirror(i) = ids[i];
        }

        for (unsigned int i = 0; i < offsets.size(); i++) {
            offsets_mirror(i) = offsets[i];
        }

        // now copy the data onto the device
        Kokkos::deep_copy(_ids, ids_mirror);
        Kokkos::deep_copy(_offsets, offsets_mirror);
    }

    Id(int n_ids, int n_values) {
        _ids = view_type("Id::id_without_init", n_ids);
        _offsets = view_type("Id::offset", n_values+1);
    }


    Id(IdConstructor constructor) 
        : Id<Layout, Space>(constructor.ids(), constructor.offsets()) 
    {}

    Id<Layout, Space> clone(){
        return Id<Layout, Space>(_ids, _offsets);
    }

    // Id<Layout, Space> clone_offsets(){
    //     view_type ids = Kokkos::View<int*>("id", static_cast<int>(_ids.size()));
    //     view_type offsets = Kokkos::View<int*> ("offset", static_cast<int>(_offsets.size())); 
    //     for (unsigned int i = 0; i < offsets.size(); i++){
    //         offsets(i) = _offsets(i);
    //     }
    //     return Id<Layout, Space>(ids, offsets);
    // }

    KOKKOS_INLINE_FUNCTION
    auto operator [] (const int i) const {
        // return the slice of _ids corresponding 
        // to the object at index i
        int first = _offsets(i);
        int last = _offsets(i+1);
        return Kokkos::subview(_ids, Kokkos::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    auto operator [] (const int i) {
        // return the slice of _ids corresponding 
        // to the object at index i
        int first = _offsets(i);
        int last = _offsets(i+1);
        return Kokkos::subview(_ids, Kokkos::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    int size() const {return _offsets.extent(0)-1;}

    bool operator == (const Id &other) const {
        for (unsigned int i = 0; i < _ids.extent(0); i++) {
            if (_ids(i) != other._ids(i)) return false;
        }
        for (unsigned int i = 0; i < _offsets.extent(0); i++) {
            if (_offsets(i) != other._offsets(i)) return false;
        }
        return true;
    }

    mirror_type host_mirror() {
        mirror_view_type ids(_ids.extent(0));
        mirror_view_type offsets(_offsets.extent(0));
        return mirror_type(ids, offsets);
    }

    template <class OtherSpace>
    void deep_copy(const Id<Layout, OtherSpace>& other){
        // std::cout << size() << " " << num_ids() << std::endl;
        // std::cout << other.size() << " " << other.num_ids() << std::endl << std::endl;
        Kokkos::deep_copy(_ids, other._ids);
        Kokkos::deep_copy(_offsets, other._offsets);
    }

    view_type ids() const {return _ids;}
    int num_ids() const {return _ids.extent(0);}
    view_type offsets() const {return _offsets;}
    // int num_offsets() const {return _offsets.extent(0);}

public:
    view_type _ids;
    view_type _offsets;
};


#endif
