#ifndef POST_H
#define POST_H

#include <CLI/CLI.hpp>

CLI::App * setup_post_cli(CLI::App& ibis);

int post(CLI::App * post_command, int argc, char* argv[]);

#endif
