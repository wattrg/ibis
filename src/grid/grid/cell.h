#ifndef CELL_H
#define CELL_H

#include <Kokkos_Core.hpp>

#include <util/field.h>
#include <util/id.h>
#include <grid/grid_io.h>
#include <grid/vertex.h>

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
struct Cells;

template <typename T,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class Space = Kokkos::DefaultExecutionSpace::memory_space>
struct CellFaces {
public:
    using array_layout = Layout;
    using memory_space = Space;
    using view_type = Kokkos::View<int*, array_layout, memory_space>;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_memory_space = typename mirror_view_type::memory_space;
    using mirror_type = CellFaces<T, array_layout, mirror_memory_space>;

    CellFaces() {}

    CellFaces(const Id<array_layout, memory_space>& interface_ids) {
        offsets_ =
            view_type("CellFaces::offsets", interface_ids.offsets().size());
        face_ids_ =
            view_type("CellFaces::face_ids", interface_ids.ids().size());
        outsigns_ =
            view_type("CellFaces::outsigns", interface_ids.ids().size());
        Kokkos::deep_copy(offsets_, interface_ids.offsets());
        Kokkos::deep_copy(face_ids_, interface_ids.ids());
    }

    CellFaces(const view_type offsets, view_type face_ids, view_type outsigns) {
        offsets_ = view_type("CellFaces::offsets", offsets.size());
        face_ids_ = view_type("CellFaces::face_ids", face_ids.size());
        outsigns_ = view_type("CellFaces::outsigns", outsigns.size());
        Kokkos::deep_copy(offsets_, offsets);
        Kokkos::deep_copy(face_ids_, face_ids);
        Kokkos::deep_copy(outsigns_, outsigns);
    }

    CellFaces(int number_cells, int number_face_ids) {
        offsets_ = view_type("CellFaces::offsets", number_cells + 1);
        face_ids_ = view_type("CellFaces::face_ids", number_face_ids);
        outsigns_ = view_type("CellFaces::outsigns", number_face_ids);
    }

    mirror_type host_mirror() const {
        return mirror_type(offsets_.size(), face_ids_.size());
    }

    bool operator==(const CellFaces& other) const {
        for (unsigned int i = 0; i < offsets_.size(); i++) {
            if (offsets_(i) != other.offsets_(i)) return false;
        }
        for (unsigned int i = 0; i < face_ids_.size(); i++) {
            if (face_ids_(i) != other.face_ids_(i)) return false;
        }
        for (unsigned int i = 0; i < outsigns_.size(); i++) {
            if (outsigns_(i) != other.outsigns_(i)) {
                for (unsigned int j = 0; j < outsigns_.size(); j++) {
                    std::cout << outsigns_(j) << " ";
                }
                std::cout << std::endl;
                for (unsigned int j = 0; j < outsigns_.size(); j++) {
                    std::cout << other.outsigns_(j) << " ";
                }
                std::cout << std::endl;
                return false;
            }
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    auto face_ids(const int i) const {
        int first = offsets_(i);
        int last = offsets_(i + 1);
        return Kokkos::subview(face_ids_, Kokkos::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    auto outsigns(const int i) const {
        int first = offsets_(i);
        int last = offsets_(i + 1);
        return Kokkos::subview(outsigns_, Kokkos::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    void set_outsign(const int i_cell, const int i_face, const int sign) {
        outsigns(i_cell)(i_face) = sign;
    }

    int num_face_ids() const { return face_ids_.extent(0); }

public:
    template <class OtherSpace>
    void deep_copy(const CellFaces<T, OtherSpace>& other) {
        Kokkos::deep_copy(offsets_, other.offsets_);
        Kokkos::deep_copy(face_ids_, other.face_ids_);
        Kokkos::deep_copy(outsigns_, other.outsigns_);
    }

public:
    view_type offsets_;
    view_type face_ids_;
    view_type outsigns_;
};

template <typename T, class ExecSpace, class Layout>
struct Cells {
public:
    using execution_space = ExecSpace;
    using memory_space = typename execution_space::memory_space;
    using array_layout = Layout;
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using host_mirror_mem_space =
        Kokkos::DefaultHostExecutionSpace::memory_space;
    using mirror_type =
        Cells<T, Kokkos::DefaultHostExecutionSpace, array_layout>;

public:
    Cells() {}

    Cells(Id<array_layout, memory_space> vertices,
          Id<array_layout, memory_space> interfaces,
          std::vector<ElemType> shapes) {
        vertex_ids_ = vertices;
        num_cells_ = shapes.size();
        faces_ = CellFaces<T, array_layout, memory_space>(interfaces);
        shape_ = Field<ElemType, array_layout, memory_space>("Cell::shape",
                                                             num_cells_);
        typename Field<ElemType, array_layout, memory_space>::mirror_type
            shape_mirror("Cell::shape", num_cells_);
        for (int i = 0; i < num_cells_; i++) {
            shape_mirror(i) = shapes[i];
        }
        shape_.deep_copy(shape_mirror);

        volume_ =
            Field<T, array_layout, memory_space>("Cell::Volume", num_cells_);
        centroid_ = Vector3s<T, array_layout, memory_space>("Cell::centroids",
                                                            num_cells_);
    }

    Cells(int num_cells, int num_vertex_ids, int num_face_ids) {
        vertex_ids_ = Id<array_layout, memory_space>(num_vertex_ids, num_cells);
        num_cells_ = num_cells;
        faces_ =
            CellFaces<T, array_layout, memory_space>(num_cells, num_face_ids);
        shape_ = Field<ElemType, array_layout, memory_space>("Cell::shape",
                                                             num_cells);
        volume_ =
            Field<T, array_layout, memory_space>("Cell::Volume", num_cells);
        centroid_ = Vector3s<T, array_layout, memory_space>("Cells::centroids",
                                                            num_cells);
    }

    mirror_type host_mirror() const {
        int num_vertex_ids = vertex_ids_.num_ids();
        int num_face_ids = faces_.num_face_ids();
        int num_cells = num_cells_;
        return mirror_type(num_cells, num_vertex_ids, num_face_ids);
    }

    template <class OtherDevice>
    void deep_copy(const Cells<T, OtherDevice, array_layout>& other) {
        vertex_ids_.deep_copy(other.vertex_ids_);
        faces_.deep_copy(other.faces_);
        shape_.deep_copy(other.shape_);
        volume_.deep_copy(other.volume_);
        centroid_.deep_copy(other.centroid_);
    }

    bool operator==(const Cells& other) const {
        return (faces_ == other.faces_) && (vertex_ids_ == other.vertex_ids_);
    }

    KOKKOS_INLINE_FUNCTION
    const Id<array_layout, memory_space>& vertex_ids() const {
        return vertex_ids_;
    }

    KOKKOS_INLINE_FUNCTION
    int size() const { return num_cells_; }

    KOKKOS_INLINE_FUNCTION
    const T& volume(const int i) const { return volume_(i); }

    KOKKOS_INLINE_FUNCTION
    const Field<T, array_layout, memory_space>& volumes() const {
        return volume_;
    }

    KOKKOS_INLINE_FUNCTION
    Field<T, array_layout, memory_space>& volumes() { return volume_; }

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T, array_layout, memory_space>& centroids() const {
        return centroid_;
    }

    KOKKOS_INLINE_FUNCTION
    Vector3s<T, array_layout, memory_space>& centroids() { return centroid_; }

    void compute_centroids(
        const Vertices<T, execution_space, array_layout>& vertices) {
        // for the moment, we're using the arithmatic average
        // of the points as the centroid. For cells that aren't
        // nicely shaped, this could be a very bad approximation
        auto centroid = centroid_;
        auto vertex_ids = vertex_ids_;
        Kokkos::parallel_for(
            "Cells::compute_centroid",
            Kokkos::RangePolicy<execution_space>(0, volume_.size()),
            KOKKOS_LAMBDA(const int i) {
                auto cell_vertices = vertex_ids[i];
                int n_vertices = cell_vertices.size();
                T x = 0.0;
                T y = 0.0;
                T z = 0.0;
                for (int v_idx = 0; v_idx < n_vertices; v_idx++) {
                    int vertex_id = cell_vertices(v_idx);
                    x += vertices.positions().x(vertex_id);
                    y += vertices.positions().y(vertex_id);
                    z += vertices.positions().z(vertex_id);
                }
                centroid.x(i) = x / n_vertices;
                centroid.y(i) = y / n_vertices;
                centroid.z(i) = z / n_vertices;
            });
    }

    void compute_volumes(
        const Vertices<T, execution_space, array_layout>& vertices) {
        // TODO: It would be nicer to move each case in the switch
        // to a function sitting somewhere else to keep the amount
        // of code in this method down, and avoid duplication with
        // computing the area of interfaces. However, this won't
        // be trivial for the GPU.
        auto volume = volume_;
        auto shape = shape_;
        auto this_vertex_ids = vertex_ids_;
        Kokkos::parallel_for(
            "Cells::compute_volume",
            Kokkos::RangePolicy<execution_space>(0, volume_.size()),
            KOKKOS_LAMBDA(const int i) {
                switch (shape(i)) {
                    case ElemType::Line:
                        printf("Invalid cell shape");
                        break;
                    case ElemType::Tri: {
                        auto vertex_ids = this_vertex_ids[i];
                        T x1 = vertices.positions().x(vertex_ids(0));
                        T x2 = vertices.positions().x(vertex_ids(1));
                        T x3 = vertices.positions().x(vertex_ids(2));
                        T y1 = vertices.positions().y(vertex_ids(0));
                        T y2 = vertices.positions().y(vertex_ids(1));
                        T y3 = vertices.positions().y(vertex_ids(2));
                        T area =
                            x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2);
                        volume(i) = 0.5 * Kokkos::fabs(area);
                        break;
                    }
                    case ElemType::Quad: {
                        auto vertex_ids = this_vertex_ids[i];
                        T x1 = vertices.positions().x(vertex_ids(0));
                        T x2 = vertices.positions().x(vertex_ids(1));
                        T x3 = vertices.positions().x(vertex_ids(2));
                        T x4 = vertices.positions().x(vertex_ids(3));
                        T y1 = vertices.positions().y(vertex_ids(0));
                        T y2 = vertices.positions().y(vertex_ids(1));
                        T y3 = vertices.positions().y(vertex_ids(2));
                        T y4 = vertices.positions().y(vertex_ids(3));
                        T area = x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1 -
                                 x2 * y1 - x3 * y2 - x4 * y3 - x1 * y4;
                        volume(i) = 0.5 * Kokkos::fabs(area);
                        break;
                    }
                    case ElemType::Hex:
                        printf("Volume of Hex not implemented");
                        break;
                    case ElemType::Wedge:
                        printf("Volume of Wedge not implemented");
                        break;
                    case ElemType::Pyramid:
                        printf("Volume of pyramid ot implemented");
                        break;
                }
            });
    }

    KOKKOS_INLINE_FUNCTION
    CellFaces<T, array_layout, memory_space> faces() const { return faces_; }

    KOKKOS_INLINE_FUNCTION
    const Field<ElemType, array_layout, memory_space>& shapes() const {
        return shape_;
    }

public:
    CellFaces<T, array_layout, memory_space> faces_;
    Id<array_layout, memory_space> vertex_ids_;
    Field<ElemType, array_layout, memory_space> shape_;
    Field<T, array_layout, memory_space> volume_;
    Vector3s<T, array_layout, memory_space> centroid_;

    int num_cells_;
};

#endif
