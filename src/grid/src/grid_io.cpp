#include <stdexcept>
#include <string>
#include <doctest/doctest.h>
#include "grid_io.h"

GridFileType file_type_from_name(std::string file_name) {
    std::size_t pos = file_name.find_last_of(".");
    std::string ext;
    if (pos != std::string::npos) {
        ext = file_name.substr(pos+1);
    }
    else {
        std::cerr << "Unable to determine file type of " 
                  << file_name 
                  << std::endl;
        throw new std::runtime_error("Unable to determine file type");
    }

    if (ext == "su2") {
        return GridFileType::Su2;
    }
    if (ext == "aeolus") {
        return GridFileType::Native;
    }
    std::cerr << "Unknown grid file type: " << ext << std::endl;
    throw new std::runtime_error("Unknown grid file type");
}

ElemType elem_type_from_su2_type(int su2_type) {
    switch (su2_type) {
        case 3:
            return ElemType::Line;
        case 5:
            return ElemType::Tri; 
        case 9:
            return ElemType::Quad;
        default:
            std::cerr << "Unknown su2 cell type: " << su2_type << std::endl;
            throw new std::runtime_error("");
    }
}

int number_vertices_from_elem_type(ElemType type) {
    switch (type) {
        case ElemType::Line:
            return 2;
        case ElemType::Tri:
            return 3;
        case ElemType::Quad:
            return 4;
        default:
            throw new std::runtime_error("");
    }
}

GridIO::GridIO(std::string file_name) {
    GridFileType type = file_type_from_name(file_name); 
    std::ifstream grid_file(file_name);
    switch (type) {
        case GridFileType::Su2:
            _read_su2_grid(grid_file);
            break;
        case GridFileType::Native:
            _read_su2_grid(grid_file);
            break;
    }
    grid_file.close();
}

void trim_whitespace(std::string &str) {
    // remove whitespace at the beginning and end of a string
    std::size_t start = str.find_first_not_of(" ");
    str = (start == std::string::npos) ? "" : str.substr(start);
    std::size_t end = str.find_last_not_of(" ");
    str = (end == std::string::npos) ? "" : str.substr(0, end+1);
}

bool starts_with(std::string &str, std::string prefix) {
    return str.rfind(prefix, 0) == 0;
}

std::string read_string(std::string str) {
    // read the value of a string of the form "NAME=VALUE", discarding the NAME
    std::size_t sep = str.find("=");
    str = str.substr(sep+1);
    trim_whitespace(str);
    return str;
}

int read_int(std::string line) {
    // read an integer from a string of the form  "NAME=int"
    line = read_string(line);
    return std::stoi(line);
}

bool get_next_line(std::ifstream &grid_file, std::string &line) {
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
    int pos = line.find(" ");
    int type_int = std::stoi(line.substr(0, pos));
    ElemType type = elem_type_from_su2_type(type_int);
    int n_vertices = number_vertices_from_elem_type(type);
    line = line.substr(pos+1, std::string::npos);
    std::vector<int> vertex_ids{};
    for (int i = 0; i < n_vertices; i++) {
        pos = line.find(" ");
        int vertex = std::stoi(line.substr(0, pos));
        line = line.substr(pos+1, std::string::npos);
        vertex_ids.push_back(vertex);
    }
    return ElemIO(vertex_ids, type);
}

Aeolus::Vector3 read_vertex(std::string line, int dim) {
    if (dim == 2) {
        int sep = line.find(" ");
        double x = std::stod(line.substr(0, sep));
        double y = std::stod(line.substr(sep));
        return Aeolus::Vector3(x, y, 0.0);
    }
    else if (dim == 3) {
        int sep = line.find(" ");
        double x = std::stod(line.substr(0, sep));
        line = line.substr(sep+1, std::string::npos);

        sep = line.find(" ");
        double y = std::stod(line.substr(0, sep));
        line = line.substr(sep+1, std::string::npos);

        sep = line.find(" ");
        double z = std::stod(line.substr(0, sep));
        return Aeolus::Vector3(x, y, z);
    }
    else {
        std::cerr << "Invalid number of dimensions in su2 file: "
                  << dim
                  << std::endl;
        throw new std::runtime_error("");
    }
}

std::pair<std::string, std::vector<ElemIO>> 
read_su2_boundary_marker(std::ifstream & grid_file, std::string &line) {
    std::string tag = read_string(line);
    get_next_line(grid_file, line);
    int n_elems = read_int(line);
    std::vector<ElemIO> elem_io {};
    for (int i = 0; i < n_elems; i++) {
        get_next_line(grid_file, line);
        elem_io.push_back(read_su2_element(line)); 
    }
    return {tag, elem_io};
}

void GridIO::_read_su2_grid(std::ifstream & grid_file) {
    // iterate through the file line by line. If we come across
    // a section heading, we read that section. If we come across
    // a line we don't know what to do with, we just ignore it.
    std::string line;
    int dimensions;
    int n_vertices;
    int n_cells;
    int n_mark;
    while (get_next_line(grid_file, line)) {

        if (starts_with(line, "NDIME")){
            dimensions = read_int(line); 
        } 

        else if (starts_with(line, "NPOIN")) {
            n_vertices = read_int(line);
            for (int vtx_i = 0; vtx_i < n_vertices; vtx_i++){
                get_next_line(grid_file, line);
                _vertices.push_back(read_vertex(line, dimensions));
            }
        }

        else if (starts_with(line, "NELEM")) {
            n_cells = read_int(line); 
            for (int cell_i = 0; cell_i < n_cells; cell_i++) {
                get_next_line(grid_file, line);
                _cells.push_back(read_su2_element(line));
            }
        }

        else if (starts_with(line, "NMARK")) {
            n_mark = read_int(line); 
            for (int mark_i = 0; mark_i < n_mark; mark_i++) {
                get_next_line(grid_file, line);
                _bcs.insert(read_su2_boundary_marker(grid_file, line)); 
            }
        }
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
    CHECK(file_type_from_name("grid.aeolus") == GridFileType::Native);
}

TEST_CASE("elem_type_from_su2_type") {
    CHECK(elem_type_from_su2_type(3) == ElemType::Line);
    CHECK(elem_type_from_su2_type(9) == ElemType::Quad);
}

TEST_CASE("number_vertices_from_elem_type") {
    CHECK(number_vertices_from_elem_type(ElemType::Line) == 2);
    CHECK(number_vertices_from_elem_type(ElemType::Tri) == 3);
    CHECK(number_vertices_from_elem_type(ElemType::Quad) == 4);
}

TEST_CASE("get_next_line") {
    std::ifstream file ("../src/grid/test/test_file.txt");
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
    CHECK(read_su2_element(line) == ElemIO({0, 1, 5, 4}, ElemType::Quad));

    line = "3 3 34";
    CHECK(read_su2_element(line) == ElemIO({3, 34}, ElemType::Line));

    line = "5 1 39 23";
    CHECK(read_su2_element(line) == ElemIO({1, 39, 23}, ElemType::Tri));
}

TEST_CASE("read_vetex") {
    std::string line = "1.0 2.0";
    CHECK(read_vertex(line, 2) == Aeolus::Vector3(1.0, 2.0, 0.0));

    line = "12.0 0.5 3.4";
    CHECK(read_vertex(line, 3) == Aeolus::Vector3(12.0, 0.5, 3.4));
}

TEST_CASE("read_su2_boundary_marker") {
    std::string line;
    std::ifstream file ("../src/grid/test/boundary_test.txt");
    std::pair<std::string, std::vector<ElemIO>> result {
        "slip_wall", {ElemIO({12, 13}, ElemType::Line), {ElemIO({13, 14, 16}, ElemType::Tri)}, {ElemIO({14, 15, 16, 1}, ElemType::Quad)}} 
    };
    get_next_line(file, line);
    CHECK(read_su2_boundary_marker(file, line) == result);
    file.close();
}
