#include <doctest/doctest.h>
#include <grid/grid_io.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <string>

ElemIO::ElemIO(const ElemIO &other) : vertex_ids_(other.vertex_ids_) {
    // vertex_ids_(other.vertex_ids_);
    cell_type_ = other.cell_type_;
    face_order_ = other.face_order_;
}

ElemIO &ElemIO::operator=(const ElemIO &other) {
    vertex_ids_.assign(other.vertex_ids_.begin(), other.vertex_ids_.end());
    cell_type_ = other.cell_type_;
    face_order_ = other.face_order_;
    return *this;
}

GridFileType file_type_from_name(std::string file_name) {
    std::size_t pos = file_name.find_last_of(".");
    std::string ext;
    if (pos != std::string::npos) {
        ext = file_name.substr(pos + 1);
    } else {
        spdlog::error("Unable to determine file type of {}", file_name);
        throw new std::runtime_error("Unable to determine file type");
    }

    if (ext == "su2") {
        return GridFileType::Su2;
    }
    if (ext == "ibis") {
        return GridFileType::Native;
    }
    spdlog::error("Unknown grid file type: {}", ext);
    throw new std::runtime_error("Unknown grid file type");
}

ElemType elem_type_from_vtk_type(size_t su2_type) {
    switch (su2_type) {
        case 3:
            return ElemType::Line;
        case 5:
            return ElemType::Tri;
        case 9:
            return ElemType::Quad;
        case 10:
            return ElemType::Tetra;
        case 12:
            return ElemType::Hex;
        case 13:
            return ElemType::Wedge;
        case 14:
            return ElemType::Pyramid;
        default:
            spdlog::error("Unknown su2 cell type: {}", su2_type);
            throw new std::runtime_error("");
    }
}

size_t vtk_type_from_elem_type(const ElemType &elem_type) {
    switch (elem_type) {
        case ElemType::Line:
            return 3;
        case ElemType::Tri:
            return 5;
        case ElemType::Quad:
            return 9;
        case ElemType::Tetra:
            return 10;
        case ElemType::Hex:
            return 12;
        case ElemType::Wedge:
            return 13;
        case ElemType::Pyramid:
            return 14;
        default:
            spdlog::error("Unkown ElemType");
            throw new std::runtime_error("");
    }
}

size_t number_vertices_from_elem_type(ElemType type) {
    switch (type) {
        case ElemType::Line:
            return 2;
        case ElemType::Tri:
            return 3;
        case ElemType::Quad:
            return 4;
        case ElemType::Tetra:
            return 4;
        case ElemType::Hex:
            return 8;
        case ElemType::Wedge:
            return 6;
        case ElemType::Pyramid:
            return 5;
        default:
            throw new std::runtime_error("");
    }
}

std::ostream &operator<<(std::ostream &file, const ElemIO &elem_io) {
    file << vtk_type_from_elem_type(elem_io.cell_type_);
    for (auto &vertex_id : elem_io.vertex_ids_) {
        file << " " << vertex_id;
    }
    return file;
}

std::ostream &operator<<(std::ostream &file, const CellMapping &map) {
    file << map.local_cell << " " << map.global_face << " " << map.local_face << " "
         << map.other_block << " " << map.other_cell;
    return file;
}

GridIO::GridIO(std::string file_name) {
    GridFileType type = file_type_from_name(file_name);
    std::ifstream grid_file(file_name);
    if (!grid_file) {
        spdlog::error("Could not find {}", file_name);
        throw new std::runtime_error("File not found");
    }
    switch (type) {
        case GridFileType::Su2:
            read_su2_grid(grid_file);
            break;
        case GridFileType::Native:
            read_su2_grid(grid_file);
            break;
    }
    grid_file.close();
    construct_faces_();
}

GridIO::GridIO(const GridIO &monolithic_grid, const std::vector<size_t> &cells_to_include,
               const std::vector<CellMapping> &&cell_mapping, size_t id)
    : id_(id), cell_mapping_(cell_mapping) {
    dim_ = monolithic_grid.dim_;
    id_ = monolithic_grid.id_;
    size_t num_cells = cells_to_include.size();
    cells_.reserve(num_cells);

    // maps global_vertex_id -> local_vertex_id
    std::unordered_map<size_t, size_t> vertex_map;
    std::unordered_map<size_t, size_t> face_map;

    for (size_t local_cell_i = 0; local_cell_i < num_cells; local_cell_i++) {
        // gather the information about the global cell
        size_t global_cell_i = cells_to_include[local_cell_i];
        ElemIO global_elem_io = monolithic_grid.cells()[global_cell_i];
        std::vector<size_t> global_vertex_ids = global_elem_io.vertex_ids();

        // gather the list of local vertices and faces for this cell
        std::vector<size_t> local_vertex_ids{};
        for (auto &global_vertex : global_vertex_ids) {
            if (vertex_map.find(global_vertex) == vertex_map.end()) {
                // we haven't encountered this vertex in this partition yet,
                // so we'll add it now
                vertex_map[global_vertex] = vertex_map.size();
                vertices_.push_back(monolithic_grid.vertices()[global_vertex]);
            }
            local_vertex_ids.push_back(vertex_map[global_vertex]);
        }

        // At this point, we have the information to build the local cell
        cells_.push_back(ElemIO(local_vertex_ids, global_elem_io.cell_type(),
                                global_elem_io.face_order()));

        // set up the faces from the global faces
        std::vector<size_t> local_face_ids{};
        std::vector<ElemIO> global_faces = global_elem_io.interfaces();
        cell_faces_ = std::vector<std::vector<size_t>> {num_cells};
        const InterfaceLookup &global_face_lookup = monolithic_grid.interface_lookup();
        InterfaceLookup local_face_lookup;
        for (auto &global_face : global_faces) {
            size_t global_face_id = global_face_lookup.id(global_face.vertex_ids());
            if (face_map.find(global_face_id) == face_map.end()) {
                // we haven't encountered this face in this partition yet,
                // so we'll add it now
                face_map[global_face_id] = face_map.size();
                faces_.push_back(global_face);
                local_face_lookup.insert(global_face.vertex_ids());
            }
            // std::cout << local_cell_i << " " << cell_faces_
            cell_faces_[local_cell_i].push_back(faces_.size() - 1);
        }
    }

    // modify the cell mappings from the global face id to the local face id
    for (CellMapping mapping : cell_mapping_) {
        mapping.local_face = face_map[mapping.global_face];
    }

    // setup the local markers for this partition
    for (const auto &[tag, global_elems] : monolithic_grid.markers_) {
        std::vector<ElemIO> local_elems;
        for (const ElemIO &global_elem : global_elems) {
            std::vector<size_t> local_vertices;
            bool elem_in_this_partition = false;
            for (const size_t &global_vertex : global_elem.vertex_ids()) {
                if (vertex_map.find(global_vertex) != vertex_map.end()) {
                    // this vertex is in this partition, so this
                    // marker should exist in this partition.
                    elem_in_this_partition = true;
                    local_vertices.push_back(vertex_map[global_vertex]);
                }
            }
            if (elem_in_this_partition) {
                local_elems.push_back(ElemIO(local_vertices, global_elem.cell_type(),
                                             global_elem.face_order()));
            }
        }
    }
}

void trim_whitespace(std::string &str) {
    // remove whitespace at the beginning and end of a string
    std::size_t start = str.find_first_not_of(" ");
    str = (start == std::string::npos) ? "" : str.substr(start);
    std::size_t end = str.find_last_not_of(" ");
    str = (end == std::string::npos) ? "" : str.substr(0, end + 1);
}

bool starts_with(std::string &str, std::string prefix) {
    return str.rfind(prefix, 0) == 0;
}

std::string read_string(std::string str) {
    // read the value of a string of the form "NAME=VALUE", discarding the NAME
    std::size_t sep = str.find("=");
    str = str.substr(sep + 1);
    trim_whitespace(str);
    return str;
}

size_t read_int(std::string line) {
    // read an integer from a string of the form  "NAME=int"
    line = read_string(line);
    return std::stoi(line);
}

bool get_next_line(std::istream &grid_file, std::string &line) {
    // read the next line of a file. Trim the whitespace from the
    // front and back of the line. Return if there was actually
    // another line in the file.
    std::getline(grid_file, line);
    if (grid_file) {
        trim_whitespace(line);
        return true;
    }
    return false;
}

ElemIO read_su2_element(std::string line) {
    size_t pos = line.find(" ");
    size_t type_int = std::stoi(line.substr(0, pos));
    ElemType type = elem_type_from_vtk_type(type_int);
    size_t n_vertices = number_vertices_from_elem_type(type);
    line = line.substr(pos + 1, std::string::npos);
    trim_whitespace(line);
    std::vector<size_t> vertex_ids{};
    for (size_t i = 0; i < n_vertices; i++) {
        pos = line.find(" ");
        size_t vertex = std::stoi(line.substr(0, pos));
        line = line.substr(pos + 1, std::string::npos);
        trim_whitespace(line);
        vertex_ids.push_back(vertex);
    }
    return ElemIO(vertex_ids, type, FaceOrder::Vtk);
}

Vector3<Ibis::real> read_vertex(std::string line, size_t dim) {
    trim_whitespace(line);
    std::string value;
    size_t sep = line.find(" ");
    value = line.substr(0, sep);
    Ibis::real x = std::stod(value);
    line = line.substr(sep + 1, std::string::npos);
    trim_whitespace(line);

    sep = line.find(" ");
    value = line.substr(0, sep);
    trim_whitespace(value);
    Ibis::real y = std::stod(value);

    if (dim == 2) {
        return Vector3(x, y, 0.0);
    } else if (dim == 3) {
        line = line.substr(sep + 1, std::string::npos);
        trim_whitespace(line);
        sep = line.find(" ");
        value = line.substr(0, sep);
        trim_whitespace(value);
        Ibis::real z = std::stod(value);
        return Vector3(x, y, z);
    } else {
        std::cerr << "Invalid number of dimensions in su2 file: " << dim << std::endl;
        throw new std::runtime_error("");
    }
}

std::pair<std::string, std::vector<ElemIO>> read_su2_marker(std::istream &grid_file,
                                                            std::string &line) {
    std::string tag = read_string(line);
    get_next_line(grid_file, line);
    size_t n_elems = read_int(line);
    std::vector<ElemIO> elem_io{};
    for (size_t i = 0; i < n_elems; i++) {
        get_next_line(grid_file, line);
        elem_io.push_back(read_su2_element(line));
    }
    return {tag, elem_io};
}

void GridIO::read_su2_grid(std::istream &grid_file) {
    // iterate through the file line by line. If we come across
    // a section heading, we read that section. If we come across
    // a line we don't know what to do with, we just ignore it.
    std::string line;
    size_t n_vertices;
    size_t n_cells;
    size_t n_mark;
    while (get_next_line(grid_file, line)) {
        if (starts_with(line, "NDIME")) {
            dim_ = read_int(line);
        }

        else if (starts_with(line, "NPOIN")) {
            n_vertices = read_int(line);
            for (size_t vtx_i = 0; vtx_i < n_vertices; vtx_i++) {
                get_next_line(grid_file, line);
                vertices_.push_back(read_vertex(line, dim_));
            }
        }

        else if (starts_with(line, "NELEM")) {
            n_cells = read_int(line);
            for (size_t cell_i = 0; cell_i < n_cells; cell_i++) {
                get_next_line(grid_file, line);
                cells_.push_back(read_su2_element(line));
            }
        }

        else if (starts_with(line, "NMARK")) {
            n_mark = read_int(line);
            for (size_t mark_i = 0; mark_i < n_mark; mark_i++) {
                get_next_line(grid_file, line);

                auto marker = read_su2_marker(grid_file, line);

                if (markers_.find(marker.first) == markers_.end()) {
                    // we haven't seen a boundary with this tag before
                    markers_.insert(marker);
                } else {
                    // we have seen a boundary with this tag before,
                    // so we append the ElemIO's that we just read
                    // to the existing ones
                    markers_[marker.first].insert(markers_[marker.first].end(),
                                                  marker.second.begin(),
                                                  marker.second.end());
                }
            }
        }
    }
}

void GridIO::write_su2_grid(std::ostream &grid_file) {
    grid_file << "NDIME= " << dim_ << "\n";

    grid_file << "NELEM= " << cells_.size() << "\n";
    for (size_t cell_i = 0; cell_i < cells_.size(); cell_i++) {
        grid_file << cells_[cell_i] << " " << cell_i << "\n";
    }

    grid_file << "NPOIN= " << vertices_.size() << "\n";
    for (size_t vertex_i = 0; vertex_i < vertices_.size(); vertex_i++) {
        grid_file << vertices_[vertex_i].pos().x << " ";
        grid_file << vertices_[vertex_i].pos().y << " ";
        if (dim_ == 3) {
            grid_file << vertices_[vertex_i].pos().z << " ";
        }
        grid_file << vertex_i << "\n";
    }

    grid_file << "NMARK= " << markers_.size() << "\n";
    for (const auto &[tag, elements] : markers_) {
        grid_file << "MARKER_TAG= " << tag << "\n";
        grid_file << "MARKER_ELEMS= " << elements.size() << "\n";
        for (const ElemIO &element : elements) {
            grid_file << element << "\n";
        }
    }
}

void GridIO::construct_faces_() {
    for (size_t cell_id = 0; cell_id < cells_.size(); cell_id++) {
        const ElemIO &cell = cells_[cell_id];
        std::vector<ElemIO> cell_interfaces = cell.interfaces();
        std::vector<size_t> this_cell_face_ids{};

        // Add each face of this cell to the list of faces
        for (size_t face_i = 0; face_i < cell_interfaces.size(); face_i++) {
            // the ID's of the vertices forming this face
            ElemIO &cell_interface = cell_interfaces[face_i];
            std::vector<size_t> face_vertices = cell_interface.vertex_ids();

            // check if this interface has already been created as part of another cell.
            // If it hasn't been created, we'll create it here
            size_t face_id = interface_lookup_.id(face_vertices);
            if (face_id == std::numeric_limits<size_t>::max()) {
                // This interface hasn't been created yet, so we do that here
                face_id = interface_lookup_.insert(face_vertices);
                faces_.push_back(ElemIO(face_vertices, cell_interface.cell_type(),
                                        cell_interface.face_order()));
            }
            this_cell_face_ids.push_back(face_id);
        }
        cell_faces_.push_back(this_cell_face_ids);
    }
}

void GridIO::read_mapped_cells(std::istream &file) {
    std::string line;
    get_next_line(file, line);
    trim_whitespace(line);
    size_t num_cells = std::stoi(line);
    cell_mapping_.reserve(num_cells);

    std::string value;
    size_t sep;
    size_t local_cell;
    size_t other_block;
    size_t other_cell;
    size_t local_face;
    size_t global_face;
    while (get_next_line(file, line)) {
        trim_whitespace(line);

        sep = line.find(" ");
        value = line.substr(0, sep);
        local_cell = std::stoi(value);
        line = line.substr(sep + 1, std::string::npos);
        trim_whitespace(line);

        sep = line.find(" ");
        value = line.substr(0, sep);
        trim_whitespace(value);
        local_face = std::stoi(value);
        line = line.substr(sep + 1, std::string::npos);
        trim_whitespace(line);

        sep = line.find(" ");
        value = line.substr(0, sep);
        trim_whitespace(value);
        global_face = std::stoi(value);
        line = line.substr(sep + 1, std::string::npos);
        trim_whitespace(line);

        sep = line.find(" ");
        value = line.substr(0, sep);
        trim_whitespace(value);
        other_block = std::stoi(value);
        line = line.substr(sep + 1, std::string::npos);
        trim_whitespace(line);

        sep = line.find(" ");
        value = line.substr(0, sep);
        trim_whitespace(value);
        other_cell = std::stoi(value);

        cell_mapping_.push_back(
            CellMapping(local_cell, local_face, global_face, other_block, other_cell));
    }
}

void GridIO::write_mapped_cells(std::ostream &file) {
    file << cell_mapping_.size() << "\n";
    for (const CellMapping &map : cell_mapping_) {
        file << map.local_cell;
        file << " ";
        file << map.local_face;
        file << " ";
        file << map.global_face;
        file << " ";
        file << map.other_block;
        file << " ";
        file << map.other_cell;
        file << "\n";
    }
}

std::vector<ElemIO> vtk_face_order(std::vector<size_t> ids, ElemType type) {
    switch (type) {
        case ElemType::Line:
            return std::vector<ElemIO>{
                ElemIO({ids[0], ids[1]}, ElemType::Line, FaceOrder::Vtk)};
        case ElemType::Tri:
            return std::vector<ElemIO>{
                ElemIO({ids[0], ids[1]}, ElemType::Line, FaceOrder::Vtk),
                ElemIO({ids[1], ids[2]}, ElemType::Line, FaceOrder::Vtk),
                ElemIO({ids[2], ids[0]}, ElemType::Line, FaceOrder::Vtk),
            };
        case ElemType::Quad:
            return std::vector<ElemIO>{
                ElemIO({ids[0], ids[1]}, ElemType::Line, FaceOrder::Vtk),
                ElemIO({ids[1], ids[2]}, ElemType::Line, FaceOrder::Vtk),
                ElemIO({ids[2], ids[3]}, ElemType::Line, FaceOrder::Vtk),
                ElemIO({ids[3], ids[0]}, ElemType::Line, FaceOrder::Vtk),
            };
        case ElemType::Tetra:
            return std::vector<ElemIO>{
                ElemIO({ids[0], ids[1], ids[2]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[0], ids[1], ids[3]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[1], ids[2], ids[3]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[0], ids[2], ids[3]}, ElemType::Tri, FaceOrder::Vtk),
            };
        case ElemType::Hex:
            return std::vector<ElemIO>{
                ElemIO({ids[0], ids[1], ids[2], ids[3]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[0], ids[1], ids[5], ids[4]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[4], ids[5], ids[6], ids[7]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[2], ids[3], ids[7], ids[6]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[0], ids[4], ids[7], ids[3]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[1], ids[5], ids[6], ids[2]}, ElemType::Quad, FaceOrder::Vtk)};
        case ElemType::Wedge:
            return std::vector<ElemIO>{
                ElemIO({ids[0], ids[1], ids[2]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[3], ids[5], ids[4]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[1], ids[4], ids[5], ids[2]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[0], ids[2], ids[5], ids[3]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[0], ids[3], ids[4], ids[1]}, ElemType::Quad, FaceOrder::Vtk)};
        case ElemType::Pyramid:
            return std::vector<ElemIO>{
                ElemIO({ids[0], ids[3], ids[2], ids[1]}, ElemType::Quad, FaceOrder::Vtk),
                ElemIO({ids[2], ids[3], ids[4]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[0], ids[4], ids[3]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[0], ids[1], ids[4]}, ElemType::Tri, FaceOrder::Vtk),
                ElemIO({ids[1], ids[2], ids[4]}, ElemType::Tri, FaceOrder::Vtk)};
        default:
            throw new std::runtime_error("Unreachable");
    }
}

std::vector<ElemIO> ElemIO::interfaces() const {
    switch (face_order_) {
        case FaceOrder::Vtk: {
            return vtk_face_order(vertex_ids_, cell_type_);
        }
        default:
            throw new std::runtime_error("Unreachable");
    }
}

#ifndef DOCTEST_CONFIG_DISABLE
TEST_CASE("trim whitespace") {
    std::string test1 = " hello world    ";
    std::string test2 = "hello world";
    std::string expected = "hello world";
    trim_whitespace(test1);
    trim_whitespace(test2);

    CHECK(test1 == expected);
    CHECK(test2 == expected);
}

TEST_CASE("starts_with") {
    std::string test1 = "NELEM= 5";
    CHECK(starts_with(test1, "NELEM") == true);
    CHECK(starts_with(test1, "asdf") == false);
    CHECK(starts_with(test1, "5") == false);
}

TEST_CASE("read string") {
    CHECK(read_string(" MARKER_TAG= inflow ") == "inflow");
    CHECK(read_string(" MARKER_TAG = inflow ") == "inflow");
    CHECK(read_string(" MARKER_TAG =inflow ") == "inflow");
}

TEST_CASE("read int") {
    CHECK(read_int("NELEM= 1") == 1);
    CHECK(read_int("NMARK = 5") == 5);
    CHECK(read_int("NMARK =6") == 6);
}

TEST_CASE("file type from name") {
    CHECK(file_type_from_name("grid.su2") == GridFileType::Su2);
    CHECK(file_type_from_name("grid.ibis") == GridFileType::Native);
}

TEST_CASE("elem_type_from_su2_type") {
    CHECK(elem_type_from_vtk_type(3) == ElemType::Line);
    CHECK(elem_type_from_vtk_type(9) == ElemType::Quad);
}

TEST_CASE("number_vertices_from_elem_type") {
    CHECK(number_vertices_from_elem_type(ElemType::Line) == 2);
    CHECK(number_vertices_from_elem_type(ElemType::Tri) == 3);
    CHECK(number_vertices_from_elem_type(ElemType::Quad) == 4);
}

TEST_CASE("get_next_line") {
    std::ifstream file("../../../src/grid/test/test_file.txt");
    std::string line;
    bool result;

    result = get_next_line(file, line);
    CHECK(line == "line 1");
    CHECK(result == true);

    result = get_next_line(file, line);
    CHECK(line == "line 2");
    CHECK(result == true);

    result = get_next_line(file, line);
    CHECK(line == "hello world");
    CHECK(result == true);

    result = get_next_line(file, line);
    CHECK(line == "line 3");
    CHECK(result == true);

    result = get_next_line(file, line);
    CHECK(result == false);
    CHECK(line == "");

    file.close();
}

TEST_CASE("read_su2_element") {
    std::string line = "9 0 1 5 4";
    CHECK(read_su2_element(line) == ElemIO({0, 1, 5, 4}, ElemType::Quad, FaceOrder::Vtk));

    line = "3 3 34";
    CHECK(read_su2_element(line) == ElemIO({3, 34}, ElemType::Line, FaceOrder::Vtk));

    line = "5 1  39 23";
    CHECK(read_su2_element(line) == ElemIO({1, 39, 23}, ElemType::Tri, FaceOrder::Vtk));

    // line = "12 1 14  15 13 19 20";
    // CHECK(read_su2_element(line) == ElemIO({1, 14, 15, 13, 19, 20}, ElemType::Hex,
    // FaceOrder::Vtk));
}

TEST_CASE("read_vetex") {
    std::string line = "1.0 2.0";
    CHECK(read_vertex(line, 2) == Vector3(1.0, 2.0, 0.0));

    line = "12.0 0.5 3.4";
    CHECK(read_vertex(line, 3) == Vector3(12.0, 0.5, 3.4));
}

TEST_CASE("read_su2_marker") {
    std::string line;
    std::ifstream file("../../../src/grid/test/boundary_test.txt");
    get_next_line(file, line);

    std::pair<std::string, std::vector<ElemIO>> result{
        "slip_wall",
        {ElemIO({12, 13}, ElemType::Line, FaceOrder::Vtk),
         ElemIO({13, 14, 16}, ElemType::Tri, FaceOrder::Vtk),
         ElemIO({14, 15, 16, 1}, ElemType::Quad, FaceOrder::Vtk)}};

    CHECK(read_su2_marker(file, line) == result);
    file.close();
}

TEST_CASE("read_su2_grid") {
    std::vector<Vertex<Ibis::real>> vertices{
        Vertex(Vector3(0.0, 0.0, 0.0)), Vertex(Vector3(1.0, 0.0, 0.0)),
        Vertex(Vector3(2.0, 0.0, 0.0)), Vertex(Vector3(3.0, 0.0, 0.0)),
        Vertex(Vector3(0.0, 1.0, 0.0)), Vertex(Vector3(1.0, 1.0, 0.0)),
        Vertex(Vector3(2.0, 1.0, 0.0)), Vertex(Vector3(3.0, 1.0, 0.0)),
        Vertex(Vector3(0.0, 2.0, 0.0)), Vertex(Vector3(1.0, 2.0, 0.0)),
        Vertex(Vector3(2.0, 2.0, 0.0)), Vertex(Vector3(3.0, 2.0, 0.0)),
        Vertex(Vector3(0.0, 3.0, 0.0)), Vertex(Vector3(1.0, 3.0, 0.0)),
        Vertex(Vector3(2.0, 3.0, 0.0)), Vertex(Vector3(3.0, 3.0, 0.0))};

    std::vector<ElemIO> cells{
        ElemIO({0, 1, 5, 4}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({1, 2, 6, 5}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({2, 3, 7, 6}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({4, 5, 9, 8}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({5, 6, 10, 9}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({6, 7, 11, 10}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({8, 9, 13, 12}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({9, 10, 14, 13}, ElemType::Quad, FaceOrder::Vtk),
        ElemIO({10, 11, 15, 14}, ElemType::Quad, FaceOrder::Vtk),
    };

    std::unordered_map<std::string, std::vector<ElemIO>> bcs{
        {"slip_wall_bottom",
         {ElemIO({0, 1}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({1, 2}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({2, 3}, ElemType::Line, FaceOrder::Vtk)}},
        {"outflow",
         {ElemIO({3, 7}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({7, 11}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({11, 15}, ElemType::Line, FaceOrder::Vtk)}},
        {"slip_wall_top",
         {ElemIO({12, 13}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({13, 14}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({14, 15}, ElemType::Line, FaceOrder::Vtk)}},
        {"inflow",
         {ElemIO({0, 4}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({4, 8}, ElemType::Line, FaceOrder::Vtk),
          ElemIO({8, 12}, ElemType::Line, FaceOrder::Vtk)}}};

    GridIO grid_io_expected = GridIO(vertices, cells, bcs);
    GridIO grid_io("../../../src/grid/test/grid.su2");
    CHECK(grid_io == grid_io_expected);
}

TEST_CASE("write_su2_grid") {
    // read the test grid
    GridIO grid_io_expected("../../../src/grid/test/grid.su2");
    std::stringstream test_file;

    // write a new grid, then read that grid
    grid_io_expected.write_su2_grid(test_file);
    GridIO grid_io;
    grid_io.read_su2_grid(test_file);

    // make sure the original grid and the re-written grid are the same
    CHECK(grid_io == grid_io_expected);
}

TEST_CASE("read_cell_mapping") {
    // read some partitioned grids
    GridIO part0("../../../src/grid/test/grid_0000.su2");
    GridIO part1("../../../src/grid/test/grid_0001.su2");

    std::ifstream mapping0_file("../../../src/grid/test/mapped_cells_0000");
    std::ifstream mapping1_file("../../../src/grid/test/mapped_cells_0001");
    part0.read_mapped_cells(mapping0_file);
    part1.read_mapped_cells(mapping1_file);

    std::vector<CellMapping> expected_map0{
        CellMapping(0, 0, 2, 1, 0),  CellMapping(0, 2, 13, 1, 4),
        CellMapping(1, 4, 6, 1, 1),  CellMapping(1, 5, 13, 1, 3),
        CellMapping(2, 9, 17, 1, 4), CellMapping(3, 10, 16, 1, 3),
    };

    std::vector<CellMapping> expected_map1{
        CellMapping(0, 2, 2, 0, 0),   CellMapping(1, 6, 6, 0, 1),
        CellMapping(3, 12, 13, 0, 1), CellMapping(3, 11, 16, 0, 3),
        CellMapping(4, 13, 11, 0, 0), CellMapping(4, 14, 17, 0, 2)};

    CHECK(part0.cell_mapping()[0] == expected_map0[0]);
    CHECK(part0.cell_mapping()[1] == expected_map0[1]);
    CHECK(part0.cell_mapping()[2] == expected_map0[2]);
    CHECK(part0.cell_mapping()[3] == expected_map0[3]);
    CHECK(part0.cell_mapping()[4] == expected_map0[4]);
    CHECK(part0.cell_mapping()[5] == expected_map0[5]);

    CHECK(part1.cell_mapping()[0] == expected_map1[0]);
    CHECK(part1.cell_mapping()[1] == expected_map1[1]);
    CHECK(part1.cell_mapping()[2] == expected_map1[2]);
    CHECK(part1.cell_mapping()[3] == expected_map1[3]);
    CHECK(part1.cell_mapping()[4] == expected_map1[4]);
    CHECK(part1.cell_mapping()[5] == expected_map1[5]);

    // CHECK(part0.cell_mapping() == expected_map0);
    // CHECK(part1.cell_mapping() == expected_map1);
}

InterfaceLookup::InterfaceLookup() {
    hash_map_ = std::unordered_map<std::string, size_t>{};
}

size_t InterfaceLookup::insert(std::vector<size_t> vertex_ids) {
    std::string hash = hash_vertex_ids(vertex_ids);
    if (contains_hash(hash)) {
        return hash_map_[hash];
    }
    size_t id = hash_map_.size();
    hash_map_.insert({hash, id});
    return id;
}

bool InterfaceLookup::contains(std::vector<size_t> vertex_ids) const {
    std::string hash = hash_vertex_ids(vertex_ids);
    return contains_hash(hash);
}

size_t InterfaceLookup::id(std::vector<size_t> vertex_ids) const {
    std::string hash = hash_vertex_ids(vertex_ids);
    if (contains_hash(hash)) {
        return hash_map_.at(hash);
    }
    return std::numeric_limits<size_t>::max();
}

bool InterfaceLookup::contains_hash(std::string hash) const {
    return hash_map_.find(hash) != hash_map_.end();
}

std::string InterfaceLookup::hash_vertex_ids(std::vector<size_t> vertex_ids) const {
    std::sort(vertex_ids.begin(), vertex_ids.end(), std::greater<size_t>());
    std::string hash_value = "";
    for (size_t i = 0; i < vertex_ids.size(); i++) {
        hash_value.append(std::to_string(vertex_ids[i]));
        hash_value.append(",");
    }
    return hash_value;
}

TEST_CASE("interface look up contains") {
    InterfaceLookup x;
    x.insert(std::vector<size_t>{0, 1});
    x.insert(std::vector<size_t>{1, 5});
    x.insert(std::vector<size_t>{5, 4});
    x.insert(std::vector<size_t>{5, 1});

    CHECK(x.contains(std::vector<size_t>{1, 0}));
    CHECK(x.contains(std::vector<size_t>{6, 1}) == false);
}

TEST_CASE("interface look up id 1") {
    InterfaceLookup x;
    x.insert(std::vector<size_t>{0, 1});
    x.insert(std::vector<size_t>{1, 5});
    x.insert(std::vector<size_t>{5, 4});
    x.insert(std::vector<size_t>{5, 1});

    CHECK(x.id(std::vector<size_t>{5, 1}) == 1);
    CHECK(x.id(std::vector<size_t>{1, 5}) == 1);
}

TEST_CASE("interface look up id 2") {
    InterfaceLookup x;
    x.insert(std::vector<size_t>{0, 1});
    x.insert(std::vector<size_t>{1, 5});
    x.insert(std::vector<size_t>{5, 4});
    x.insert(std::vector<size_t>{5, 1});

    CHECK(x.id(std::vector<size_t>{6, 1}) == -1);
}

TEST_CASE("interface look up") {
    InterfaceLookup x;
    x.insert(std::vector<size_t>{0, 1});
    x.insert(std::vector<size_t>{1, 5});
    x.insert(std::vector<size_t>{5, 4});
    x.insert(std::vector<size_t>{4, 0});
    x.insert(std::vector<size_t>{1, 2});
    x.insert(std::vector<size_t>{2, 6});
    x.insert(std::vector<size_t>{6, 5});
    x.insert(std::vector<size_t>{5, 1});
    x.insert(std::vector<size_t>{2, 3});
    x.insert(std::vector<size_t>{3, 7});
    x.insert(std::vector<size_t>{7, 6});
    x.insert(std::vector<size_t>{6, 2});

    CHECK(x.id(std::vector<size_t>{0, 1}) == 0);
    CHECK(x.id(std::vector<size_t>{1, 5}) == 1);
    CHECK(x.id(std::vector<size_t>{5, 4}) == 2);
    CHECK(x.id(std::vector<size_t>{4, 0}) == 3);
    CHECK(x.id(std::vector<size_t>{1, 2}) == 4);
    CHECK(x.id(std::vector<size_t>{2, 6}) == 5);
    CHECK(x.id(std::vector<size_t>{6, 5}) == 6);
    CHECK(x.id(std::vector<size_t>{2, 3}) == 7);
    CHECK(x.id(std::vector<size_t>{3, 7}) == 8);
    CHECK(x.id(std::vector<size_t>{7, 6}) == 9);
}

#endif  // DOCTEST_CONFIG_DISABLE
