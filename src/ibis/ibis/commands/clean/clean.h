#ifndef CLEAN_H
#define CLEAN_H

#include <CLI/CLI.hpp>

void setup_clean_cli(CLI::App& ibis);

int clean(int argc, char* arvg[]);

#endif
