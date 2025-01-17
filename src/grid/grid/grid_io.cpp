#include <doctest/doctest.h>
#include <grid/grid_io.h>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <string>

ElemIO::ElemIO(const ElemIO& other) : vertex_ids_(other.vertex_ids_) {
    // vertex_ids_(other.vertex_ids_};
    cell_type_ = other.cell_type_;
    face_order_ = other.face_order_;
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
        case FaceOrder::Vtk:
            return vtk_face_order(vertex_ids_, cell_type_);
        default:
            throw new std::runtime_error("Unreachable");
    }
}

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
