#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nbody/config.hpp"
#include "nbody/cpu_solver.hpp"
#include "nbody/cuda_solver.hpp"
#include "nbody/io.hpp"

namespace {

struct Arguments {
    std::filesystem::path input;
    std::filesystem::path config;
    std::filesystem::path output;
    std::string backend = "cpu";
    int block_size = 128;
};

void print_usage(const char* program) {
    std::cerr << "用法: " << program
              << " --input particles.txt --config simulation.cfg --output results/run"
              << " [--backend cpu|cuda-naive|cuda-tiled]"
              << " [--block-size 64|128|256|512]\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--backend") {
            if (++i >= argc) throw std::runtime_error("--backend 缺少值");
            arguments.backend = argv[i];
            if (arguments.backend != "cpu" &&
                arguments.backend != "cuda-naive" &&
                arguments.backend != "cuda-tiled") {
                throw std::runtime_error(
                    "--backend 只支持 cpu、cuda-naive 或 cuda-tiled");
            }
            continue;
        }
        if (option == "--block-size") {
            if (++i >= argc) throw std::runtime_error("--block-size 缺少值");
            std::size_t used = 0;
            try {
                arguments.block_size = std::stoi(argv[i], &used);
            } catch (const std::exception&) {
                throw std::runtime_error("--block-size 必须是整数");
            }
            if (used != std::string(argv[i]).size() ||
                (arguments.block_size != 64 && arguments.block_size != 128 &&
                 arguments.block_size != 256 && arguments.block_size != 512)) {
                throw std::runtime_error("--block-size 只支持 64、128、256 或 512");
            }
            continue;
        }
        if (option != "--input" && option != "--config" && option != "--output") {
            throw std::runtime_error("未知参数: " + option);
        }
        if (++i >= argc) throw std::runtime_error(option + " 缺少值");
        if (option == "--input") arguments.input = argv[i];
        if (option == "--config") arguments.config = argv[i];
        if (option == "--output") arguments.output = argv[i];
    }
    if (arguments.input.empty() || arguments.config.empty() || arguments.output.empty()) {
        throw std::runtime_error("必须提供 --input、--config 和 --output");
    }
    return arguments;
}

}  // namespace

int main(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    try {
        const auto wall_begin = Clock::now();
        const Arguments arguments = parse_arguments(argc, argv);
        if (std::filesystem::exists(arguments.output)) {
            throw std::runtime_error("输出目录已存在，请使用新目录: " +
                                     arguments.output.string());
        }
        const nbody::Config config = nbody::read_config(arguments.config);
        const auto particles = nbody::read_particles(arguments.input);
        std::filesystem::create_directories(arguments.output);
        nbody::CpuRunResult result;
        std::string precision;
        if (arguments.backend == "cpu") {
            result = nbody::simulate_cpu(particles, config);
            precision = "fp64";
        } else if (arguments.backend == "cuda-naive") {
            result = nbody::simulate_cuda_naive(particles, config,
                                                arguments.block_size);
            precision = "fp32";
        } else {
            result = nbody::simulate_cuda_tiled(particles, config,
                                                arguments.block_size);
            precision = "fp32";
        }
        const double before_write_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - wall_begin).count();
        nbody::write_run_outputs(arguments.output, arguments.input, config, result,
                                 before_write_ms, arguments.backend, precision);
        std::cout << arguments.backend << ' ' << precision << " 模拟完成\n"
                  << "粒子: " << particles.size() << ", 步数: " << config.num_steps
                  << ", 积分器: " << nbody::to_string(config.integrator) << '\n'
                  << "力计算: " << result.force_ms << " ms, 模拟: "
                  << result.simulation_ms << " ms\n"
                  << "输出目录: " << arguments.output << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "错误: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
}
