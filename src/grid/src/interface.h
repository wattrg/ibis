#ifndef INTERFACE_H
#define INTERFACE_H

#include <unordered_map>
#include "../../util/src/id.h"
#include "../../util/src/field.h"
#include "../../util/src/vector3.h"
#include "Kokkos_Core_fwd.hpp"
#include "Kokkos_Macros.hpp"
#include "grid_io.h"
#include "vertex.h"
#include "cell.h"


template <typename T, 
          class ExecSpace=Kokkos::DefaultExecutionSpace,
          class Layout=Kokkos::DefaultExecutionSpace::array_layout>
struct Interfaces {
public:
    using execution_space = ExecSpace;
    using memory_space = typename execution_space::memory_space;
    using array_layout = Layout;
    using host_execution_space = Kokkos::DefaultHostExecutionSpace;
    using host_mirror_mem_space = Kokkos::DefaultHostExecutionSpace::memory_space;
    using mirror_type = Interfaces<T, Kokkos::DefaultHostExecutionSpace, array_layout>;

public:
    Interfaces () {}

    Interfaces(IdConstructor ids, std::vector<ElemType> shapes){
        vertex_ids_ = Id<array_layout, memory_space>(ids);
        size_ = vertex_ids_.size();
        shape_ = Field<ElemType, array_layout, memory_space>("Interface::shape", shapes.size());
        auto shape_mirror = shape_.host_mirror();
        for (int i = 0; i < size_; i++) {
            shape_mirror(i) = shapes[i];
        }
        shape_.deep_copy(shape_mirror);

        // geometry
        norm_ = Vector3s<T, array_layout, memory_space>("Interface::norm", size_);
        tan1_ = Vector3s<T, array_layout, memory_space>("Interface::tan1", size_);
        tan2_ = Vector3s<T, array_layout, memory_space>("Interface::tan2", size_);
        area_ = Field<T, array_layout, memory_space>("Interface::area", size_);
        centre_ = Vector3s<T, array_layout, memory_space>("Interface::centre", size_);

        // set left and right cells to -1 to indicate they haven't
        // been connected up to any cells yet
        left_cells_ = Field<int, array_layout, memory_space>("Interface::left", size_);
        right_cells_ = Field<int, array_layout, memory_space>("Interface::right", size_);
        left_cells_.deep_copy(-1);
        right_cells_.deep_copy(-1);
    }

    Interfaces(int num_interfaces, int num_vertex_ids) {
        vertex_ids_ = Id<array_layout, memory_space>(num_vertex_ids, num_interfaces);
        size_ = num_interfaces;
        shape_ = Field<ElemType, array_layout, memory_space>("Interface::shape", size_);
        norm_ = Vector3s<T, array_layout, memory_space>("Interface::norm", size_);
        tan1_ = Vector3s<T, array_layout, memory_space>("Interface::tan1", size_);
        tan2_ = Vector3s<T, array_layout, memory_space>("Interface::tan2", size_);
        centre_ = Vector3s<T, array_layout, memory_space>("Interface::centre", size_);
        area_ = Field<T, array_layout, memory_space>("Interface::area", size_);
        left_cells_ = Field<int, array_layout, memory_space>("Interface::left", size_);
        right_cells_ = Field<int, array_layout, memory_space>("Interface::right", size_);
    }

    mirror_type host_mirror() const {
        return mirror_type(size_, vertex_ids_.num_ids());
    }

    template <class OtherSpace>
    void deep_copy(const Interfaces<T, OtherSpace, array_layout>& other) {
        vertex_ids_.deep_copy(other.vertex_ids_);
        shape_.deep_copy(other.shape_);
        norm_.deep_copy(other.norm_);
        tan1_.deep_copy(other.tan1_);
        tan2_.deep_copy(other.tan2_);
        centre_.deep_copy(other.centre_);
        area_.deep_copy(other.area_);
        left_cells_.deep_copy(other.left_cells_);
        right_cells_.deep_copy(other.right_cells_);
    }

    bool operator == (const Interfaces &other) const {
        return vertex_ids_ == other.vertex_ids_;
    }

    KOKKOS_INLINE_FUNCTION 
    Id<array_layout, memory_space> &vertex_ids() {return vertex_ids_;}

    KOKKOS_INLINE_FUNCTION 
    const Id<array_layout, memory_space> &vertex_ids() const {return vertex_ids_;}

    KOKKOS_INLINE_FUNCTION
    const Field<T, array_layout, memory_space> &area() const {return area_;}

    KOKKOS_INLINE_FUNCTION
    T& area(int i) const {return area_(i);}

    KOKKOS_INLINE_FUNCTION
    T& area(int i) {return area_(i);}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T, array_layout, memory_space>& norm() {return norm_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T, array_layout, memory_space>& norm() const {return norm_;}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T, array_layout, memory_space>& tan1() {return tan1_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T, array_layout, memory_space>& tan1() const {return tan1_;}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T, array_layout, memory_space>& tan2() {return tan2_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T, array_layout, memory_space>& tan2() const {return tan2_;}

    KOKKOS_INLINE_FUNCTION
    Vector3s<T, array_layout, memory_space>& centre() {return centre_;}

    KOKKOS_INLINE_FUNCTION
    const Vector3s<T, array_layout, memory_space>& centre() const {return centre_;}

    KOKKOS_INLINE_FUNCTION
    void attach_cell_left(const int cell_id,  const int face_id) const {
        left_cells_(face_id) = cell_id;
    }

    KOKKOS_INLINE_FUNCTION
    void attach_cell_right(const int cell_id, const int face_id) const {
        right_cells_(face_id) = cell_id;
    }

    KOKKOS_INLINE_FUNCTION
    int left_cell(const int face_id) const {return left_cells_(face_id);}

    KOKKOS_INLINE_FUNCTION
    int right_cell(const int face_id) const {return right_cells_(face_id);}

    KOKKOS_INLINE_FUNCTION
    int size() const {return vertex_ids_.size();}

    void compute_orientations(Vertices<T, execution_space> vertices){
        // set the face tangents in parallel
        auto this_norm = norm_;
        auto this_tan1 = tan1_;
        auto this_tan2 = tan2_;
        auto this_vertex_ids = vertex_ids_;
        auto shape = shape_;
        Kokkos::parallel_for("Interfaces::compute_orientations", 
                             Kokkos::RangePolicy<execution_space>(0, norm_.size()), 
                             KOKKOS_LAMBDA (const int i){
            auto vertex_ids = this_vertex_ids[i];
            T x0 = vertices.positions().x(vertex_ids(0));
            T x1 = vertices.positions().x(vertex_ids(1));
            T y0 = vertices.positions().y(vertex_ids(0));
            T y1 = vertices.positions().y(vertex_ids(1));
            T z0 = vertices.positions().z(vertex_ids(0));
            T z1 = vertices.positions().z(vertex_ids(1));
            T ilength = 1./Kokkos::sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0));
            this_tan1.x(i) = ilength * (x1 - x0);
            this_tan1.y(i) = ilength * (y1 - y0);
            this_tan1.z(i) = ilength * (z1 - z0);

            switch (shape(i)) {
                case ElemType::Line: 
                {
                    auto vertex_ids = this_vertex_ids[i];
                    this_tan2.x(i) = 0.0;
                    this_tan2.y(i) = 0.0;
                    this_tan2.z(i) = 1.0;
                    break;
                }
                case ElemType::Tri: 
                    printf("Tri faces not implemented yet");
                    break;
                case ElemType::Quad:
                    printf("Quad faces not implemented yet");
                    break;
                default:
                    printf("Invalid interface shape");
                    break;
            }
            cross(this_tan1, this_tan2, this_norm, i);
        });
    }

    void compute_areas(Vertices<T, execution_space> vertices){
        auto this_area = area_;
        auto shape = shape_;
        auto this_vertex_ids = vertex_ids_;
        Kokkos::parallel_for("Interfaces::compute_areas", 
                             Kokkos::RangePolicy<execution_space>(0, area_.size()), 
                             KOKKOS_LAMBDA (const int i) {
            switch (shape(i)) {
                case ElemType::Line: {
                    auto vertex_ids = this_vertex_ids[i];
                    T x1 = vertices.positions().x(vertex_ids(0));
                    T x2 = vertices.positions().x(vertex_ids(1));
                    T y1 = vertices.positions().y(vertex_ids(0));
                    T y2 = vertices.positions().y(vertex_ids(1));
                    this_area(i) = Kokkos::sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
                    break;
                }
                case ElemType::Tri: {
                    auto vertex_ids = this_vertex_ids[i];
                    T x1 = vertices.positions().x(vertex_ids(0));
                    T x2 = vertices.positions().x(vertex_ids(1));
                    T x3 = vertices.positions().y(vertex_ids(2));
                    T y1 = vertices.positions().y(vertex_ids(0));
                    T y2 = vertices.positions().z(vertex_ids(1));
                    T y3 = vertices.positions().z(vertex_ids(2));
                    T area = 0.5*Kokkos::fabs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2));
                    this_area(i) = area;
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
                    T area = x1*y2 + x2*y3 + x3*y4 + x4*y1 - x2*y1 - x3*y2 - x4*y3 - x1*y4;
                    this_area(i) = 0.5 * Kokkos::fabs(area);
                    break;
                }
                case ElemType::Hex: {
                    printf("Invalid interface"); 
                    break;
                }
                case ElemType::Wedge: {
                    printf("Invalid interface"); 
                    break;
                }
                case ElemType::Pyramid: {
                    printf("Invalid interface"); 
                    break;
                }
            }
        });
    }

    void compute_centres(Vertices<T, execution_space> vertices) {
        auto centre = centre_;
        auto vertex_ids = vertex_ids_;
        Kokkos::parallel_for("Interfaces::compute_centres", 
                             Kokkos::RangePolicy<execution_space>(0, centre_.size()), 
                             KOKKOS_LAMBDA (const int face_i){
            auto face_vertices = vertex_ids[face_i]; 
            T x = 0.0;
            T y = 0.0;
            T z = 0.0;
            unsigned int num_vertices = face_vertices.size();
            for (unsigned int vtx_i = 0; vtx_i < num_vertices; vtx_i++) {
                int vtx_id = face_vertices[vtx_i];
                x += vertices.positions().x(vtx_id);  
                y += vertices.positions().y(vtx_id);
                z += vertices.positions().z(vtx_id);
            }
            centre.x(face_i) = x / num_vertices;
            centre.y(face_i) = y / num_vertices;
            centre.z(face_i) = z / num_vertices;
        });
    }


public:
    int size_;
    // the id's of the vertices forming each interface
    Id<array_layout, memory_space> vertex_ids_;

    // the cells to the left/right of the interface
    Field<int, array_layout, memory_space> left_cells_;
    Field<int, array_layout, memory_space> right_cells_;

    // geometric data
    Field<T, array_layout, memory_space> area_;
    Field<ElemType, array_layout, memory_space> shape_;
    Vector3s<T, array_layout, memory_space> norm_;
    Vector3s<T, array_layout, memory_space> tan1_;
    Vector3s<T, array_layout, memory_space> tan2_;
    Vector3s<T, array_layout, memory_space> centre_;
};


// Efficient look-up of interface ID 
// from the index of the vertices
// forming the interface
struct InterfaceLookup {
public:
    InterfaceLookup();

    int insert(std::vector<int> vertex_ids);
    bool contains(std::vector<int> vertex_ids);
    int id(std::vector<int> vertex_ids); 

private:
    std::unordered_map<std::string, int> hash_map_;

    std::string hash_vertex_ids(std::vector<int> vertex_ids);
    bool contains_hash(std::string hash);
};

#endif
