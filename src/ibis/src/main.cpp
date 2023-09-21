#include <iostream>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "ibis_version_info.h"
#include "commands/prep.h"
#include "commands/clean.h"
#include "commands/run.h"


static std::string HELP = 
    "ibis usage:\n"
    "ibis command [options]\n"
    "\n"
    "Available commands:\n"
    "    help: write this help message\n"
    "    prep [filename]: prepare a simulation given a python input script\n"
    "    run: run a simulation"
    "    clean: clean a directory of generated files";



void print_header() {
    spdlog::info("ibis - cfd solver");
    spdlog::info("git branch: {}",Ibis::GIT_BRANCH);
    spdlog::info("git commit: {}-{}", Ibis::GIT_COMMIT_HASH, Ibis::GIT_CLEAN_STATUS); 
    spdlog::info("revision date: {}", Ibis::GIT_COMMIT_DATE);
    spdlog::info("build date: {}", Ibis::IBIS_BUILD_DATE);
}

int main(int argc, char* argv[]) {
    std::vector<spdlog::sink_ptr> logs;
    auto console_log = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_log->set_pattern("[%^%l%$] %v");
    console_log->set_level(spdlog::level::info);
    logs.push_back(console_log);

    auto debug_log = std::make_shared<spdlog::sinks::basic_file_sink_mt>("log/log");
    debug_log->set_level(spdlog::level::debug);
    logs.push_back(debug_log);
    auto logger = std::make_shared<spdlog::logger>("logger", begin(logs), end(logs));

    spdlog::set_default_logger(logger);
    print_header(); 

    doctest::Context ctx;

    if (argc < 2) {
        spdlog::error("Not enough arguments provided\n");
        return 1;
    }

    std::string command = argv[1];
    
    if (command == "help") {
        std::cout << HELP;
        return 0;
    }

    else if (command == "prep") {
        return prep(argc, argv); 
    }

    else if (command == "clean") {
        return clean(argc, argv);
    }

    else if (command == "run") {
        return run(argc, argv);
    }

    else {
        std::cerr << "Unknown command: " << command << std::endl;
        std::cerr << "For help, use `ibis help`" << std::endl;
    }
}





  
// typedef Kokkos::View<double*[3]> FieldTest;
// int test(int argc, char* argv[]) {
//     Kokkos::initialize(argc, argv);
//
//     // write hello a few times
//     Kokkos::parallel_for("Hello world", 15, KOKKOS_LAMBDA (const int i){
//         printf("Hello from i = %i\n", i);
//     });
//
//     // add up square numbers in parallel
//     int sum = 0;
//     const int n = 10;
//     Kokkos::parallel_reduce("reduction", n, KOKKOS_LAMBDA (const int i, int& lsum) {
//         lsum += i*i;
//     }, sum);
//     printf("The sum of the first %i square numbers is %i\n", n-1, sum);
//
//     {
//         // play around with a views
//         FieldTest a ("A", n);
//         Kokkos::parallel_for("View", n, KOKKOS_LAMBDA (const int i) {
//             a(i, 0) = 1.0 * i;
//             a(i, 1) = 1.0 * i * i;
//             a(i, 2) = 1.0 * i * i * i;
//         });
//         double view_sum = 0.0;
//         Kokkos::parallel_reduce ("Reduction", n, KOKKOS_LAMBDA (const int i, double& update) {
//             update += a(i, 0) * a(i,1) / (a(i,2) + 0.1);
//         }, view_sum);
//         printf("Result: %f\n", view_sum);
//     }
//
//     {
//         GasStates<double> gs = GasStates<double>(5);
//         std::cout << "Built some gas states!" << std::endl;
//         printf("initial gs energy = %f\n", gs.energy(2));
//     }
//
//     Kokkos::finalize();
//     return 0;
// }
//
