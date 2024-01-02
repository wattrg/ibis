#ifndef CELL_H
#define CELL_H

#include <grid/grid_io.h>
#include <grid/vertex.h>
#include <util/field.h>
#include <util/geom.h>
#include <util/ragged_array.h>

#include <Kokkos_Core.hpp>

template <typename T, class ExecSpace = Kokkos::DefaultExecutionSpace,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout>
struct Cells;

template <typename T,
          class Layout = Kokkos::DefaultExecutionSpace::array_layout,
          class ExecSpace = Kokkos::DefaultExecutionSpace>
struct CellFaces {
public:
    using array_layout = Layout;
    using memory_space = typename ExecSpace::memory_space;
    using execution_space = ExecSpace;
    using view_type = Kokkos::View<size_t*, array_layout, memory_space>;
    using signed_view_type = Kokkos::View<int*, array_layout, memory_space>;
    using mirror_view_type = typename view_type::host_mirror_type;
    using mirror_layout = typename mirror_view_type::array_layout;
    using mirror_memory_space = typename mirror_view_type::memory_space;
    using mirror_exec_space = Kokkos::DefaultHostExecutionSpace;
    using mirror_type = CellFaces<T, array_layout, mirror_exec_space>;

    CellFaces() {}

    CellFaces(const Ibis::RaggedArray<size_t, array_layout, execution_space>&
                  interface_ids) {
        offsets_ =
            view_type("CellFaces::offsets", interface_ids.offsets().size());
        face_ids_ =
            view_type("CellFaces::face_ids", interface_ids.data().size());
        outsigns_ = signed_view_type("CellFaces::outsigns",
                                     interface_ids.data().size());
        Kokkos::deep_copy(offsets_, interface_ids.offsets());
        Kokkos::deep_copy(face_ids_, interface_ids.data());
    }

    CellFaces(const view_type offsets, view_type face_ids,
              signed_view_type outsigns)
        : offsets_(offsets), face_ids_(face_ids), outsigns_(outsigns) {}

    CellFaces(size_t number_cells, size_t number_face_ids) {
        offsets_ = view_type("CellFaces::offsets", number_cells + 1);
        face_ids_ = view_type("CellFaces::face_ids", number_face_ids);
        outsigns_ = signed_view_type("CellFaces::outsigns", number_face_ids);
    }

    mirror_type host_mirror() const {
        auto offsets = Kokkos::create_mirror_view(offsets_);
        auto face_ids = Kokkos::create_mirror_view(face_ids_);
        auto outsigns = Kokkos::create_mirror_view(outsigns_);
        return mirror_type(offsets, face_ids, outsigns);
    }

    bool operator==(const CellFaces& other) const {
        for (size_t i = 0; i < offsets_.size(); i++) {
            if (offsets_(i) != other.offsets_(i)) return false;
        }
        for (size_t i = 0; i < face_ids_.size(); i++) {
            if (face_ids_(i) != other.face_ids_(i)) return false;
        }
        for (size_t i = 0; i < outsigns_.size(); i++) {
            if (outsigns_(i) != other.outsigns_(i)) {
                for (size_t j = 0; j < outsigns_.size(); j++) {
                    std::cout << outsigns_(j) << " ";
                }
                std::cout << std::endl;
                for (size_t j = 0; j < outsigns_.size(); j++) {
                    std::cout << other.outsigns_(j) << " ";
                }
                std::cout << std::endl;
                return false;
            }
        }
        return true;
    }

    KOKKOS_INLINE_FUNCTION
    auto face_ids(const size_t i) const {
        size_t first = offsets_(i);
        size_t last = offsets_(i + 1);
        return Kokkos::subview(face_ids_, Kokkos::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    auto outsigns(const size_t i) const {
        size_t first = offsets_(i);
        size_t last = offsets_(i + 1);
        return Kokkos::subview(outsigns_, Kokkos::make_pair(first, last));
    }

    KOKKOS_INLINE_FUNCTION
    void set_outsign(const size_t i_cell, const size_t i_face, const int sign) {
        outsigns(i_cell)(i_face) = sign;
    }

    size_t num_face_ids() const { return face_ids_.extent(0); }

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
    signed_view_type outsigns_;
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

    Cells(Ibis::RaggedArray<size_t, array_layout, execution_space> vertices,
          Ibis::RaggedArray<size_t, array_layout, execution_space> interfaces,
          std::vector<ElemType> shapes, size_t num_valid_cells,
          size_t num_ghost_cells) {
        vertex_ids_ = vertices;
        num_valid_cells_ = num_valid_cells;
        num_ghost_cells_ = num_ghost_cells;
        faces_ = CellFaces<T, array_layout, execution_space>(interfaces);

        // this initially sets the incorrect neighbour cells, so we
        // have to be careful to overwrite them properly
        neighbour_cells_ =
            Ibis::RaggedArray<size_t, array_layout, execution_space>(
                interfaces);

        shape_ = Field<ElemType, array_layout, memory_space>("Cell::shape",
                                                             num_valid_cells_);
        typename Field<ElemType, array_layout, memory_space>::mirror_type
            shape_mirror("Cell::shape", num_valid_cells_);
        for (size_t i = 0; i < num_valid_cells_; i++) {
            shape_mirror(i) = shapes[i];
        }
        shape_.deep_copy(shape_mirror);

        size_t total_cells = num_valid_cells_ + num_ghost_cells_;
        volume_ =
            Field<T, array_layout, memory_space>("Cell::Volume", total_cells);
        centroid_ = Vector3s<T, array_layout, memory_space>("Cell::centroids",
                                                            total_cells);
    }

    Cells(Ibis::RaggedArray<size_t, array_layout, execution_space> vertices,
          CellFaces<T, array_layout, execution_space> faces,
          Ibis::RaggedArray<size_t, array_layout, execution_space> neighbours,
          Field<ElemType, array_layout, memory_space> shapes,
          Field<T, array_layout, memory_space> volume,
          Vector3s<T, array_layout, memory_space> centroid,
          size_t num_valid_cells, size_t num_ghost_cells)
        : faces_(faces),
          vertex_ids_(vertices),
          neighbour_cells_(neighbours),
          shape_(shapes),
          volume_(volume),
          centroid_(centroid),
          num_valid_cells_(num_valid_cells),
          num_ghost_cells_(num_ghost_cells) {}

    mirror_type host_mirror() const {
        auto vertices = vertex_ids_.host_mirror();
        auto faces = faces_.host_mirror();
        auto neighbours = neighbour_cells_.host_mirror();
        auto shapes = shape_.host_mirror();
        auto volume = volume_.host_mirror();
        auto centroid = centroid_.host_mirror();
        return mirror_type(vertices, faces, neighbours, shapes, volume,
                           centroid, num_valid_cells_, num_ghost_cells_);
    }

    template <class OtherDevice>
    void deep_copy(const Cells<T, OtherDevice, array_layout>& other) {
        vertex_ids_.deep_copy(other.vertex_ids_);
        faces_.deep_copy(other.faces_);
        shape_.deep_copy(other.shape_);
        volume_.deep_copy(other.volume_);
        centroid_.deep_copy(other.centroid_);
        neighbour_cells_.deep_copy(other.neighbour_cells_);
    }

    bool operator==(const Cells& other) const {
        return (faces_ == other.faces_) && (vertex_ids_ == other.vertex_ids_);
    }

    KOKKOS_INLINE_FUNCTION
    const Ibis::RaggedArray<size_t, array_layout, execution_space>& vertex_ids()
        const {
        return vertex_ids_;
    }

    KOKKOS_INLINE_FUNCTION
    size_t num_valid_cells() const { return num_valid_cells_; }

    KOKKOS_INLINE_FUNCTION
    size_t num_ghost_cells() const { return num_ghost_cells_; }

    KOKKOS_INLINE_FUNCTION
    size_t num_total_cells() const {
        return num_valid_cells_ + num_ghost_cells_;
    }

    KOKKOS_INLINE_FUNCTION
    const T& volume(const size_t i) const { return volume_(i); }

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
            Kokkos::RangePolicy<execution_space>(0, num_valid_cells_),
            KOKKOS_LAMBDA(const size_t i) {
                auto cell_vertices = vertex_ids(i);
                size_t n_vertices = cell_vertices.size();
                T x = 0.0;
                T y = 0.0;
                T z = 0.0;
                for (size_t v_idx = 0; v_idx < n_vertices; v_idx++) {
                    size_t vertex_id = cell_vertices(v_idx);
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
        auto volume = volume_;
        auto shape = shape_;
        auto this_vertex_ids = vertex_ids_;
        Kokkos::parallel_for(
            "Cells::compute_volume",
            Kokkos::RangePolicy<execution_space>(0, num_valid_cells_),
            KOKKOS_LAMBDA(const size_t i) {
                switch (shape(i)) {
                    case ElemType::Line:
                        printf("Invalid cell shape: Line");
                        break;
                    case ElemType::Tri: {
                        auto vertex_ids = this_vertex_ids(i);
                        volume(i) = Ibis::area_of_triangle(
                            vertices.positions(), vertex_ids(0), vertex_ids(1),
                            vertex_ids(2));
                        break;
                    }
                    case ElemType::Quad: {
                        auto vertex_ids = this_vertex_ids(i);
                        volume(i) = Ibis::area_of_quadrilateral(
                            vertices.positions(), vertex_ids(0), vertex_ids(1),
                            vertex_ids(2), vertex_ids(3));
                        break;
                    }
                    case ElemType::Hex:
                        printf("Volume of Hex not implemented");
                        break;
                    case ElemType::Wedge:
                        printf("Volume of Wedge not implemented");
                        break;
                    case ElemType::Pyramid:
                        printf("Volume of pyramid not implemented");
                        break;
                }
            });
    }

    KOKKOS_INLINE_FUNCTION
    CellFaces<T, array_layout, execution_space> faces() const { return faces_; }

    KOKKOS_INLINE_FUNCTION
    const Field<ElemType, array_layout, memory_space>& shapes() const {
        return shape_;
    }

    KOKKOS_INLINE_FUNCTION
    void set_cell_neighbour(size_t cell_i, size_t face_i,
                            size_t neighbour) const {
        neighbour_cells_(cell_i, face_i) = neighbour;
    }

    KOKKOS_INLINE_FUNCTION
    size_t neighbour_cells(const size_t cell_i, const size_t face_i) const {
        return neighbour_cells_(cell_i, face_i);
    }

    KOKKOS_INLINE_FUNCTION
    auto neighbour_cells(const size_t cell_i) const {
        return neighbour_cells_(cell_i);
    }

    KOKKOS_INLINE_FUNCTION
    auto neighbour_cells() const { return neighbour_cells_; }

public:
    CellFaces<T, array_layout, execution_space> faces_;
    Ibis::RaggedArray<size_t, array_layout, execution_space> vertex_ids_;
    Ibis::RaggedArray<size_t, array_layout, execution_space> neighbour_cells_;
    Field<ElemType, array_layout, memory_space> shape_;
    Field<T, array_layout, memory_space> volume_;
    Vector3s<T, array_layout, memory_space> centroid_;

    size_t num_valid_cells_;
    size_t num_ghost_cells_;
};

#endif
