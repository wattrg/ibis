#ifndef PREP_H
#define PREP_H

#include <CLI/CLI.hpp>

void setup_prep_cli(CLI::App& ibis);

int prep(int argc, char* argv[]);

#endif
