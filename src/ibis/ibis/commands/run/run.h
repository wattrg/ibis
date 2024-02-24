#ifndef RUN_H
#define RUN_H

#include <CLI/CLI.hpp>

void setup_run_cli(CLI::App& ibis);

int run(int argc, char* argv[]);

#endif
