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
#include "commands/post.h"


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
    else if (command == "post") {
        return post(argc, argv);
    }

    else {
        std::cerr << "Unknown command: " << command << std::endl;
        std::cerr << "For help, use `ibis help`" << std::endl;
    }
}
