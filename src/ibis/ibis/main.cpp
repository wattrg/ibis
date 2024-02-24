#include <CLI/App.hpp>
#include <iostream>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <ibis/commands/clean/clean.h>
#include <ibis/commands/post_commands/post.h>
#include <ibis/commands/prep/prep.h>
#include <ibis/commands/run/run.h>
// #include <ibis_version_info.h>
#include <runtime_dirs.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

static std::string HELP =
    "ibis usage:\n"
    "ibis command [options]\n"
    "\n"
    "Available commands:\n"
    "    help: write this help message\n"
    "    prep [filename]: prepare a simulation given a python input script\n"
    "    run: run a simulation"
    "    clean: clean a directory of generated files";

int main(int argc, char* argv[]) {
    // set up the logger
    std::vector<spdlog::sink_ptr> logs;
    auto console_log = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_log->set_pattern("[%^%l%$] %v");
    console_log->set_level(spdlog::level::info);
    logs.push_back(console_log);

    auto debug_log =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>("log/log");
    debug_log->set_level(spdlog::level::debug);
    logs.push_back(debug_log);
    auto logger =
        std::make_shared<spdlog::logger>("logger", begin(logs), end(logs));

    spdlog::set_default_logger(logger);

    // doctest requires every executable which links to libraries
    // using doctest (which our libraries do), to have a doctest
    // context. We build that here, but it never gets executed.
    // This should have very little overhead on this executable.
    doctest::Context ctx;

    // set up the command line interface
    CLI::App ibis{"compressible computational fluid dynamics"};
    ibis.require_subcommand(1);
    setup_clean_cli(ibis);
    setup_prep_cli(ibis);
    setup_run_cli(ibis);
    setup_post_cli(ibis);

    CLI11_PARSE(ibis, argc, argv);

    if (ibis.got_subcommand("clean")) {
        return clean(argc, argv);
    }
    if (ibis.got_subcommand("prep")) {
        return prep(argc, argv);
    }
    if (ibis.got_subcommand("run")) {
        return run(argc, argv);
    }
    if (ibis.got_subcommand("post")) {
        return post(argc, argv);
    }
    // if (argc < 2) {
    //     spdlog::error("Not enough arguments provided\n");
    //     return 1;
    // }
    //
    // std::string command = argv[1];
    //
    // if (command == "help") {
    //     std::cout << HELP;
    //     return 0;
    // }
    //
    // else if (command == "prep") {
    //     return prep(argc, argv);
    // }
    //
    // else if (command == "clean") {
    //     return clean(argc, argv);
    // }
    //
    // else if (command == "run") {
    //     run(argc, argv);
    //     return 0;
    // } else if (command == "post") {
    //     return post(argc, argv);
    // }
    //
    // else {
    //     std::cerr << "Unknown command: " << command << std::endl;
    //     std::cerr << "For help, use `ibis help`" << std::endl;
    // }
}
