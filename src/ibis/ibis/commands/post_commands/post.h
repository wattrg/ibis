#ifndef POST_H
#define POST_H

#include <CLI/CLI.hpp>

void setup_post_cli(CLI::App& ibis);

int post(int argc, char* argv[]);

#endif
