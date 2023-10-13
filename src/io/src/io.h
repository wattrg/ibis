#ifndef FV_IO_H
#define FV_IO_H

#include <string>
#include <spdlog/spdlog.h>
#include "../../src/gas/src/flow_state.h"
#include "../../src/grid/src/grid.h"

enum class FlowFormat {
    Native, Vtk
};

template <typename T>
class FVInput {
public:
    virtual ~FVInput(){}

    virtual int read(FlowStates<T>& fs, const GridBlock<T>& grid, std::string dir)=0;
};

template <typename T>
class FVOutput {
public:
    virtual ~FVOutput(){}

    virtual int write(const FlowStates<T>& fs, const GridBlock<T>& grid, std::string plot_dir, std::string time_dir, double time) = 0;

    virtual void write_coordinating_file(std::string plot_dir)=0;
};

template <typename T>
class FVIO {
public:
    FVIO(FlowFormat input_format, FlowFormat output_format, std::string input_dir, std::string output_dir, int time_index);

    FVIO(FlowFormat input, FlowFormat output);

    FVIO(FlowFormat input, FlowFormat output, std::string input_dir, std::string output_dir);

    FVIO();

    int read(FlowStates<T>& fs, const GridBlock<T>& grid, int time_idx);

    int write(const FlowStates<T>& flow_state, const GridBlock<T>& grid, double time);
    void write_coordinating_file();

private:
    std::unique_ptr<FVInput<T>> input_;
    std::unique_ptr<FVOutput<T>> output_;
    int time_index_;
    std::string input_dir_;
    std::string output_dir_;
};

#endif
