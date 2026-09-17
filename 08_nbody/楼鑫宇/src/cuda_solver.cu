#include "nbody/cuda_solver.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace nbody {
namespace {

using Clock = std::chrono::steady_clock;

enum class ForceKernel { Naive, Tiled, MixedHalf2 };

void check_cuda(cudaError_t status, const char* expression, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA 错误，行 ") +
                                 std::to_string(line) + "，" + expression +
                                 ": " + cudaGetErrorString(status));
    }
}

#define NBODY_CUDA_CHECK(expression) \
    check_cuda((expression), #expression, __LINE__)

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        NBODY_CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() { return data_; }
    const T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

struct HalfPositionBuffers {
    explicit HalfPositionBuffers(std::size_t particle_count)
        : pair_count((particle_count + 1) / 2),
          x(pair_count),
          y(pair_count),
          z(pair_count),
          mass(pair_count) {}

    std::size_t pair_count;
    DeviceBuffer<__half2> x;
    DeviceBuffer<__half2> y;
    DeviceBuffer<__half2> z;
    DeviceBuffer<__half2> mass;
};

class EventPair {
public:
    EventPair() {
        NBODY_CUDA_CHECK(cudaEventCreate(&start_));
        try {
            NBODY_CUDA_CHECK(cudaEventCreate(&stop_));
        } catch (...) {
            cudaEventDestroy(start_);
            throw;
        }
    }

    ~EventPair() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    cudaEvent_t start() const { return start_; }
    cudaEvent_t stop() const { return stop_; }

private:
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

__global__ void force_naive_kernel(const float4* positions,
                                   float3* accelerations,
                                   int particle_count,
                                   float gravitational_constant,
                                   float softening_squared) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;

    const float4 target = positions[i];
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;
    for (int j = 0; j < particle_count; ++j) {
        if (j == i) continue;
        const float4 source = positions[j];
        const float dx = source.x - target.x;
        const float dy = source.y - target.y;
        const float dz = source.z - target.z;
        const float distance_squared =
            dx * dx + dy * dy + dz * dz + softening_squared;
        const float inverse_distance = rsqrtf(distance_squared);
        const float scale = gravitational_constant * source.w *
                            inverse_distance * inverse_distance * inverse_distance;
        ax += dx * scale;
        ay += dy * scale;
        az += dz * scale;
    }
    accelerations[i] = make_float3(ax, ay, az);
}

__global__ void force_tiled_kernel(const float4* positions,
                                   float3* accelerations,
                                   int particle_count,
                                   float gravitational_constant,
                                   float softening_squared) {
    extern __shared__ float4 tile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid_target = i < particle_count;
    const float4 target = valid_target ? positions[i] : make_float4(0, 0, 0, 0);
    float ax = 0.0f;
    float ay = 0.0f;
    float az = 0.0f;

    for (int tile_begin = 0; tile_begin < particle_count;
         tile_begin += blockDim.x) {
        const int source_index = tile_begin + threadIdx.x;
        tile[threadIdx.x] = source_index < particle_count
                                ? positions[source_index]
                                : make_float4(0, 0, 0, 0);
        __syncthreads();

        if (valid_target) {
            const int remaining = particle_count - tile_begin;
            const int tile_count = remaining < blockDim.x ? remaining : blockDim.x;
            for (int source_in_tile = 0; source_in_tile < tile_count;
                 ++source_in_tile) {
                const int j = tile_begin + source_in_tile;
                if (j == i) continue;
                const float4 source = tile[source_in_tile];
                const float dx = source.x - target.x;
                const float dy = source.y - target.y;
                const float dz = source.z - target.z;
                const float distance_squared =
                    dx * dx + dy * dy + dz * dz + softening_squared;
                const float inverse_distance = rsqrtf(distance_squared);
                const float scale = gravitational_constant * source.w *
                                    inverse_distance * inverse_distance *
                                    inverse_distance;
                ax += dx * scale;
                ay += dy * scale;
                az += dz * scale;
            }
        }
        __syncthreads();
    }

    if (valid_target) accelerations[i] = make_float3(ax, ay, az);
}

__global__ void pack_positions_half2_kernel(const float4* positions,
                                            __half2* x,
                                            __half2* y,
                                            __half2* z,
                                            __half2* mass,
                                            int particle_count,
                                            int pair_count) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair >= pair_count) return;

    const int first = pair * 2;
    const int second = first + 1;
    const float4 first_position = positions[first];
    const float4 second_position =
        second < particle_count ? positions[second] : make_float4(0, 0, 0, 0);
    x[pair] = __floats2half2_rn(first_position.x, second_position.x);
    y[pair] = __floats2half2_rn(first_position.y, second_position.y);
    z[pair] = __floats2half2_rn(first_position.z, second_position.z);
    mass[pair] = __floats2half2_rn(first_position.w, second_position.w);
}

__global__ void force_mixed_half2_kernel(const float4* positions,
                                         const __half2* x,
                                         const __half2* y,
                                         const __half2* z,
                                         const __half2* mass,
                                         float3* accelerations,
                                         int particle_count,
                                         int pair_count,
                                         float gravitational_constant,
                                         float softening_squared) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;

    const float4 target = positions[i];
    const __half2 target_x = __float2half2_rn(target.x);
    const __half2 target_y = __float2half2_rn(target.y);
    const __half2 target_z = __float2half2_rn(target.z);
    const __half2 gravity = __float2half2_rn(gravitational_constant);
    const __half2 epsilon_squared = __float2half2_rn(softening_squared);
    float ax_even = 0.0f;
    float ax_odd = 0.0f;
    float ay_even = 0.0f;
    float ay_odd = 0.0f;
    float az_even = 0.0f;
    float az_odd = 0.0f;

    for (int pair = 0; pair < pair_count; ++pair) {
        const __half2 dx = __hsub2(x[pair], target_x);
        const __half2 dy = __hsub2(y[pair], target_y);
        const __half2 dz = __hsub2(z[pair], target_z);
        __half2 distance_squared = __hfma2(dx, dx, epsilon_squared);
        distance_squared = __hfma2(dy, dy, distance_squared);
        distance_squared = __hfma2(dz, dz, distance_squared);
        __half2 source_mass = mass[pair];
        const int first_source = pair * 2;
        if (first_source == i) {
            distance_squared = __halves2half2(
                __float2half(1.0f), __high2half(distance_squared));
            source_mass = __halves2half2(__float2half(0.0f),
                                         __high2half(source_mass));
        } else if (first_source + 1 == i) {
            distance_squared = __halves2half2(
                __low2half(distance_squared), __float2half(1.0f));
            source_mass = __halves2half2(__low2half(source_mass),
                                         __float2half(0.0f));
        }
        const __half2 inverse_distance = h2rsqrt(distance_squared);
        __half2 scale = __hmul2(gravity, source_mass);
        scale = __hmul2(scale, inverse_distance);
        scale = __hmul2(scale, inverse_distance);
        scale = __hmul2(scale, inverse_distance);

        const float2 x_contribution = __half22float2(__hmul2(dx, scale));
        const float2 y_contribution = __half22float2(__hmul2(dy, scale));
        const float2 z_contribution = __half22float2(__hmul2(dz, scale));
        ax_even += x_contribution.x;
        ax_odd += x_contribution.y;
        ay_even += y_contribution.x;
        ay_odd += y_contribution.y;
        az_even += z_contribution.x;
        az_odd += z_contribution.y;
    }
    accelerations[i] = make_float3(ax_even + ax_odd, ay_even + ay_odd,
                                   az_even + az_odd);
}

__global__ void euler_update_kernel(float4* positions,
                                    float3* velocities,
                                    const float3* accelerations,
                                    int particle_count,
                                    float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    float4 position = positions[i];
    float3 velocity = velocities[i];
    const float3 acceleration = accelerations[i];
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;
    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;
    velocity.z += acceleration.z * dt;
    positions[i] = position;
    velocities[i] = velocity;
}

__global__ void kick_drift_kernel(float4* positions,
                                  float3* velocities,
                                  const float3* accelerations,
                                  int particle_count,
                                  float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    float4 position = positions[i];
    float3 velocity = velocities[i];
    const float3 acceleration = accelerations[i];
    const float half_dt = 0.5f * dt;
    velocity.x += acceleration.x * half_dt;
    velocity.y += acceleration.y * half_dt;
    velocity.z += acceleration.z * half_dt;
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;
    positions[i] = position;
    velocities[i] = velocity;
}

__global__ void final_kick_kernel(float3* velocities,
                                  const float3* accelerations,
                                  int particle_count,
                                  float dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) return;
    float3 velocity = velocities[i];
    const float3 acceleration = accelerations[i];
    const float half_dt = 0.5f * dt;
    velocity.x += acceleration.x * half_dt;
    velocity.y += acceleration.y * half_dt;
    velocity.z += acceleration.z * half_dt;
    velocities[i] = velocity;
}

float launch_force(const float4* positions,
                   float3* accelerations,
                   int particle_count,
                   float gravitational_constant,
                   float softening_squared,
                   int grid_size,
                   int block_size,
                   const EventPair& events,
                   ForceKernel force_kernel,
                   HalfPositionBuffers* half_positions) {
    NBODY_CUDA_CHECK(cudaEventRecord(events.start()));
    if (force_kernel == ForceKernel::Naive) {
        force_naive_kernel<<<grid_size, block_size>>>(
            positions, accelerations, particle_count, gravitational_constant,
            softening_squared);
    } else if (force_kernel == ForceKernel::Tiled) {
        const std::size_t shared_bytes =
            static_cast<std::size_t>(block_size) * sizeof(float4);
        force_tiled_kernel<<<grid_size, block_size, shared_bytes>>>(
            positions, accelerations, particle_count, gravitational_constant,
            softening_squared);
    } else {
        if (half_positions == nullptr) {
            throw std::logic_error("混合精度缓冲区未初始化");
        }
        const int pair_count = static_cast<int>(half_positions->pair_count);
        const int pair_grid_size = (pair_count + block_size - 1) / block_size;
        pack_positions_half2_kernel<<<pair_grid_size, block_size>>>(
            positions, half_positions->x.get(), half_positions->y.get(),
            half_positions->z.get(), half_positions->mass.get(), particle_count,
            pair_count);
        NBODY_CUDA_CHECK(cudaGetLastError());
        force_mixed_half2_kernel<<<grid_size, block_size>>>(
            positions, half_positions->x.get(), half_positions->y.get(),
            half_positions->z.get(), half_positions->mass.get(), accelerations,
            particle_count, pair_count, gravitational_constant,
            softening_squared);
    }
    NBODY_CUDA_CHECK(cudaGetLastError());
    NBODY_CUDA_CHECK(cudaEventRecord(events.stop()));
    NBODY_CUDA_CHECK(cudaEventSynchronize(events.stop()));
    float elapsed_ms = 0.0f;
    NBODY_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, events.start(), events.stop()));
    return elapsed_ms;
}

void record_snapshot(const std::vector<float4>& positions,
                     std::size_t record_index,
                     std::size_t record_count,
                     std::vector<float>& trajectory) {
    for (std::size_t particle = 0; particle < positions.size(); ++particle) {
        const float4 position = positions[particle];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z)) {
            throw std::runtime_error("CUDA 轨迹包含非有限坐标");
        }
        const std::size_t offset =
            (particle * record_count + record_index) * 3;
        trajectory[offset] = position.x;
        trajectory[offset + 1] = position.y;
        trajectory[offset + 2] = position.z;
    }
}

}  // namespace

static CpuRunResult simulate_cuda(const std::vector<Particle>& initial_particles,
                                  const Config& config,
                                  int block_size,
                                  ForceKernel force_kernel,
                                  const RunOptions& options) {
    if (initial_particles.empty()) throw std::invalid_argument("至少需要一个粒子");
    if (initial_particles.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error("CUDA 粒子数量超出 int 范围");
    }
    if (block_size < 1 || block_size > 1024) {
        throw std::invalid_argument("CUDA block size 必须在 1 到 1024 之间");
    }

    const std::size_t particle_count = initial_particles.size();
    const int n = static_cast<int>(particle_count);
    const int grid_size = (n + block_size - 1) / block_size;
    std::vector<float4> host_positions(particle_count);
    std::vector<float3> host_velocities(particle_count);

    CpuRunResult result;
    result.record_enabled = options.record_enabled;
    result.diagnostics_enabled = options.diagnostics_enabled;
    result.particles.resize(particle_count);
    for (std::size_t i = 0; i < particle_count; ++i) {
        const Particle& source = initial_particles[i];
        const float4 position = make_float4(
            static_cast<float>(source.position.x),
            static_cast<float>(source.position.y),
            static_cast<float>(source.position.z),
            static_cast<float>(source.mass));
        const float3 velocity = make_float3(
            static_cast<float>(source.velocity.x),
            static_cast<float>(source.velocity.y),
            static_cast<float>(source.velocity.z));
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(position.w) ||
            !std::isfinite(velocity.x) || !std::isfinite(velocity.y) ||
            !std::isfinite(velocity.z) || !(position.w > 0.0f)) {
            throw std::runtime_error("初态无法用 CUDA FP32 表示");
        }
        host_positions[i] = position;
        host_velocities[i] = velocity;
        result.particles[i] = {
            {position.x, position.y, position.z},
            {velocity.x, velocity.y, velocity.z},
            position.w,
        };
    }

    if (options.record_enabled) {
        result.recorded_steps = make_recorded_steps(config.num_steps,
                                                     config.record_interval);
    }
    const std::size_t record_count = result.recorded_steps.size();
    if (options.record_enabled &&
        record_count > std::numeric_limits<std::size_t>::max() /
                           particle_count / 3) {
        throw std::overflow_error("CUDA 轨迹大小溢出");
    }
    if (options.record_enabled) {
        result.trajectory.resize(particle_count * record_count * 3);
        record_snapshot(host_positions, 0, record_count, result.trajectory);
    }
    if (options.diagnostics_enabled) {
        result.initial_diagnostics = compute_diagnostics(
            result.particles, config.gravitational_constant, config.softening);
    }

    DeviceBuffer<float4> device_positions(particle_count);
    DeviceBuffer<float3> device_velocities(particle_count);
    DeviceBuffer<float3> device_accelerations(particle_count);
    std::unique_ptr<HalfPositionBuffers> half_positions;
    if (force_kernel == ForceKernel::MixedHalf2) {
        half_positions = std::make_unique<HalfPositionBuffers>(particle_count);
    }
    NBODY_CUDA_CHECK(cudaMemcpy(device_positions.get(), host_positions.data(),
                                particle_count * sizeof(float4),
                                cudaMemcpyHostToDevice));
    NBODY_CUDA_CHECK(cudaMemcpy(device_velocities.get(), host_velocities.data(),
                                particle_count * sizeof(float3),
                                cudaMemcpyHostToDevice));

    const float gravitational_constant =
        static_cast<float>(config.gravitational_constant);
    const float softening = static_cast<float>(config.softening);
    const float softening_squared = softening * softening;
    const float dt = static_cast<float>(config.dt);
    if (!std::isfinite(gravitational_constant) ||
        !std::isfinite(softening_squared) || !std::isfinite(dt) ||
        !(gravitational_constant > 0.0f) || !(softening > 0.0f) ||
        !(softening_squared > 0.0f) || !(dt > 0.0f)) {
        throw std::runtime_error("配置参数无法用 CUDA FP32 表示");
    }
    if (force_kernel == ForceKernel::MixedHalf2) {
        const auto half_roundtrip = [](float value) {
            return __half2float(__float2half_rn(value));
        };
        const float half_gravity = half_roundtrip(gravitational_constant);
        const float half_softening_squared = half_roundtrip(softening_squared);
        if (!std::isfinite(half_gravity) || !(half_gravity > 0.0f) ||
            !std::isfinite(half_softening_squared) ||
            !(half_softening_squared > 0.0f)) {
            throw std::runtime_error("G 或 softening^2 无法用 CUDA FP16 表示");
        }
        for (const float4& position : host_positions) {
            const float half_mass = half_roundtrip(position.w);
            const float half_gravity_mass =
                half_roundtrip(half_gravity * half_mass);
            if (!std::isfinite(half_roundtrip(position.x)) ||
                !std::isfinite(half_roundtrip(position.y)) ||
                !std::isfinite(half_roundtrip(position.z)) ||
                !std::isfinite(half_mass) || !(half_mass > 0.0f) ||
                !std::isfinite(half_gravity_mass) ||
                !(half_gravity_mass > 0.0f)) {
                throw std::runtime_error("初态位置、质量或 G×质量无法用 CUDA FP16 表示");
            }
        }
    }

    EventPair force_events;
    const auto simulation_begin = Clock::now();
    result.force_ms += launch_force(
        device_positions.get(), device_accelerations.get(), n,
        gravitational_constant, softening_squared, grid_size, block_size,
        force_events, force_kernel, half_positions.get());
    result.force_evaluations = 1;
    std::size_t next_record = 1;

    for (int step = 1; step <= config.num_steps; ++step) {
        if (config.integrator == Integrator::Euler) {
            euler_update_kernel<<<grid_size, block_size>>>(
                device_positions.get(), device_velocities.get(),
                device_accelerations.get(), n, dt);
            NBODY_CUDA_CHECK(cudaGetLastError());
            if (step < config.num_steps) {
                result.force_ms += launch_force(
                    device_positions.get(), device_accelerations.get(), n,
                    gravitational_constant, softening_squared, grid_size,
                    block_size, force_events, force_kernel,
                    half_positions.get());
                ++result.force_evaluations;
            }
        } else {
            kick_drift_kernel<<<grid_size, block_size>>>(
                device_positions.get(), device_velocities.get(),
                device_accelerations.get(), n, dt);
            NBODY_CUDA_CHECK(cudaGetLastError());
            result.force_ms += launch_force(
                device_positions.get(), device_accelerations.get(), n,
                gravitational_constant, softening_squared, grid_size,
                block_size, force_events, force_kernel, half_positions.get());
            ++result.force_evaluations;
            final_kick_kernel<<<grid_size, block_size>>>(
                device_velocities.get(), device_accelerations.get(), n, dt);
            NBODY_CUDA_CHECK(cudaGetLastError());
        }

        if (options.record_enabled && next_record < record_count &&
            result.recorded_steps[next_record] == step) {
            NBODY_CUDA_CHECK(cudaMemcpy(
                host_positions.data(), device_positions.get(),
                particle_count * sizeof(float4), cudaMemcpyDeviceToHost));
            record_snapshot(host_positions, next_record, record_count,
                            result.trajectory);
            ++next_record;
        }
    }

    NBODY_CUDA_CHECK(cudaDeviceSynchronize());
    result.simulation_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - simulation_begin)
            .count();
    NBODY_CUDA_CHECK(cudaMemcpy(host_positions.data(), device_positions.get(),
                                particle_count * sizeof(float4),
                                cudaMemcpyDeviceToHost));
    NBODY_CUDA_CHECK(cudaMemcpy(host_velocities.data(), device_velocities.get(),
                                particle_count * sizeof(float3),
                                cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < particle_count; ++i) {
        const float4 position = host_positions[i];
        const float3 velocity = host_velocities[i];
        if (!std::isfinite(position.x) || !std::isfinite(position.y) ||
            !std::isfinite(position.z) || !std::isfinite(velocity.x) ||
            !std::isfinite(velocity.y) || !std::isfinite(velocity.z)) {
            throw std::runtime_error("CUDA 最终状态包含非有限数值");
        }
        result.particles[i] = {
            {position.x, position.y, position.z},
            {velocity.x, velocity.y, velocity.z},
            position.w,
        };
    }
    if (options.diagnostics_enabled) {
        result.final_diagnostics = compute_diagnostics(
            result.particles, config.gravitational_constant, config.softening);
    }
    return result;
}

CpuRunResult simulate_cuda_naive(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size,
                                 const RunOptions& options) {
    return simulate_cuda(initial_particles, config, block_size,
                         ForceKernel::Naive, options);
}

CpuRunResult simulate_cuda_tiled(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size,
                                 const RunOptions& options) {
    return simulate_cuda(initial_particles, config, block_size,
                         ForceKernel::Tiled, options);
}

CpuRunResult simulate_cuda_mixed(const std::vector<Particle>& initial_particles,
                                 const Config& config,
                                 int block_size,
                                 const RunOptions& options) {
    return simulate_cuda(initial_particles, config, block_size,
                         ForceKernel::MixedHalf2, options);
}

}  // namespace nbody
