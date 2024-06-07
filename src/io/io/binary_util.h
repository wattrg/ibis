#ifndef BINARY_UTIL
#define BINARY_UTIL

#include <fstream>
#include <string>
#include <vector>

template <typename T>
void write_binary(std::ofstream& f, T& value) {
    f.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
void read_binary(std::ifstream& f, T& value) {
    f.read(reinterpret_cast<char*>(&value), sizeof(T));
}

#endif
