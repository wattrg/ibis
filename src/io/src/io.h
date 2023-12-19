#ifndef FV_IO_H
#define FV_IO_H

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>

#include <gas/flow_state.h>
#include <gas/gas_model.h>
#include "../../src/grid/src/grid.h"

using json = nlohmann::json;

enum class FlowFormat { Native, Vtk };

template <typename T>
class FVInput {
public:
    virtual ~FVInput() {}

    virtual int read(typename FlowStates<T>::mirror_type& fs,
                     const typename GridBlock<T>::mirror_type& grid,
                     const IdealGas<T>& gas_model, std::string dir,
                     json& meta_data) = 0;
};

template <typename T>
class FVOutput {
public:
    virtual ~FVOutput() {}

    virtual int write(const typename FlowStates<T>::mirror_type& fs,
                      const typename GridBlock<T>::mirror_type& grid,
                      const IdealGas<T>& gas_model, std::string plot_dir,
                      std::string time_dir, double time) = 0;

    virtual void write_coordinating_file(std::string plot_dir) = 0;
};

template <typename T>
class FVIO {
public:
    // constructors
    FVIO(FlowFormat input_format, FlowFormat output_format,
         std::string input_dir, std::string output_dir, int time_index);

    FVIO(FlowFormat input, FlowFormat output);

    FVIO(FlowFormat input, FlowFormat output, std::string input_dir,
         std::string output_dir);

    FVIO(int time_index);

    FVIO();

    // read a flow state
    int read(FlowStates<T>& fs, const GridBlock<T>& grid,
             const IdealGas<T>& gas_model, json& meta_data, int time_idx);

    // write a flow state
    int write(const FlowStates<T>& flow_state, const GridBlock<T>& grid,
              const IdealGas<T>& gas_model, double time);
    void write_coordinating_file();

private:
    std::unique_ptr<FVInput<T>> input_;
    std::unique_ptr<FVOutput<T>> output_;
    int time_index_;
    std::string input_dir_;
    std::string output_dir_;
};

#endif
