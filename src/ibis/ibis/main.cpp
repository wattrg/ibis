#include <ibis_version.h>

#include <CLI/App.hpp>
#include <iostream>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>
#include <ibis/commands/clean/clean.h>
#include <ibis/commands/post_commands/plot.h>
#include <ibis/commands/post_commands/plot_residuals.h>
#include <ibis/commands/post_commands/post.h>
#include <ibis/commands/prep/prep.h>
#include <ibis/commands/run/run.h>
#include <io/io.h>
// #include <ibis_version_info.h>
#include <runtime_dirs.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>

int cli(int argc, char* argv[]) {
    // set up the command line interface
    CLI::App ibis{"compressible computational fluid dynamics"};
    ibis.failure_message(CLI::FailureMessage::help);
    bool print_version = false;
    ibis.add_flag("-v,--version", print_version, "print the version");

    CLI::App* clean_command = ibis.add_subcommand("clean", "clean the simulation");
    CLI::App* prep_command = ibis.add_subcommand("prep", "prepare the simulation");
    CLI::App* run_command = ibis.add_subcommand("run", "run the simulation");

    CLI::App* post_command = ibis.add_subcommand("post", "post-process the simulation");
    post_command->require_subcommand(1);

    CLI::App* plot_command = post_command->add_subcommand(
        "plot", "write simulatioin files to visualisation format");
    std::map<std::string, FlowFormat> format_map{
        {"vtk-binary", FlowFormat::VtkBinary},
        {"vtk-text", FlowFormat::VtkText},
    };
    FlowFormat format = FlowFormat::VtkBinary;
    plot_command->add_option("-f,--format", format, "File format")
        ->option_text("format (default: vtk-binary)")
        ->transform(CLI::CheckedTransformer(format_map, CLI::ignore_case));

    std::vector<std::string> extra_vars;
    plot_command->add_option("--add", extra_vars, "Extra variables to add to plot")
        ->delimiter(',')
        ->type_name("str");

    CLI::App* plot_residuals_command =
        post_command->add_subcommand("plot_residuals", "plot simulation residuals");

    // parse the command line
    CLI11_PARSE(ibis, argc, argv);

    // execute whatever the user asked for
    if (print_version) {
        std::cout << Ibis::IBIS_VERSION << std::endl;
        return 0;
    }
    if (ibis.got_subcommand(clean_command)) {
        return clean(argc, argv);
    } else if (ibis.got_subcommand(prep_command)) {
        return prep(argc, argv);
    } else if (ibis.got_subcommand(run_command)) {
        return run(argc, argv);
    } else if (ibis.got_subcommand("post")) {
        if (post_command->got_subcommand(plot_command)) {
            return plot(format, extra_vars, argc, argv);
        } else if (post_command->got_subcommand(plot_residuals_command)) {
            return plot_residuals();
        }
    } else {
        spdlog::error("Nothing to do. Try `ibis --help`");
    }
    return 1;
}

int main(int argc, char* argv[]) {
    // set up the logger
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

    // doctest requires every executable which links to libraries
    // using doctest (which our libraries do), to have a doctest
    // context. We build that here, but it never gets executed.
    // This should have very little overhead on this executable.
    doctest::Context ctx;

    // run the command line interface
    return cli(argc, argv);
}
