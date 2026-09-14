# N 体引力模拟与可视化（楼鑫宇）

当前阶段完成了 CPU FP64 参考后端和 CUDA FP32 朴素直接求和后端。两个后端支持题目规定的粒子输入、配置文件、Euler/Leapfrog 积分和二进制轨迹输出，并有固定小规模逐粒子对照测试。

## 已实现

- 严格解析 `x y z vx vy vz mass` 粒子文件。
- 严格解析 `dt`、`num_steps`、`record_interval`、`G`、`softening`、`integrator`。
- CPU FP64 全粒子直接求和，复杂度为 O(N²)。
- CUDA FP32 朴素全粒子直接求和，每个线程负责一个目标粒子。
- 显式 Euler 和 Leapfrog KDK，默认使用 Leapfrog。
- 输出粒子优先的 float32 轨迹 `[P,R,3]`。
- 输出最终状态、初末守恒量、性能与元数据。
- C++ 单元测试、CPU/CUDA 逐粒子对照、双体轨道验证和固定随机种子的星团生成脚本。

当前 CUDA 版本是正确性基线，尚未使用共享内存分块；CPU FP64 继续作为 GPU 版本的高精度参考。

## 构建

```bash
source /data/nbody-dev/activate-nbody.sh
cd /data/nbody-project/08_nbody/楼鑫宇
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

测试集合包含 CPU 核心公式、输入解析、轨迹哨兵往返、Leapfrog 步长收敛，以及 129 粒子 `cuda-naive` 与 CPU FP64 的逐粒子最终状态和轨迹对照。

## CUDA 朴素版

```bash
./build/nbody \
  --backend cuda-naive \
  --block-size 128 \
  --input data/two_body.txt \
  --config configs/two_body.cfg \
  --output results/two_body_cuda_naive

python scripts/validate_two_body.py results/two_body_cuda_naive
```

`cuda-naive` 在 GPU 上以 FP32 保存位置、速度、质量和加速度，输出元数据会明确记录 `backend=cuda-naive` 与 `precision=fp32`。

## 双体实验

```bash
./build/nbody \
  --backend cpu \
  --input data/two_body.txt \
  --config configs/two_body.cfg \
  --output results/two_body_cpu

python scripts/validate_two_body.py results/two_body_cpu
```

输出目录已存在时程序会拒绝覆盖，请换一个目录或在确认不需要旧结果后手动处理。

## 4096 粒子 CPU 基线

```bash
python scripts/generate_cluster.py --n 4096 --seed 42 --output data/cluster_4096.txt

./build/nbody \
  --backend cpu \
  --input data/cluster_4096.txt \
  --config configs/cpu_4096.cfg \
  --output results/cluster_4096_cpu_1000
```

服务器上的首轮实测数据、实验条件和结果解读见
[`docs/cpu-baseline.md`](docs/cpu-baseline.md)。后续比较 CUDA 后端时必须复用相同初态、配置和记录策略。

## 4096 粒子短实验

这个实验只用 10 步确认 CPU O(N²) 路径在基础规模上可运行，不代表题目要求的 1000 步最终成绩。

```bash
python scripts/generate_cluster.py --n 4096 --seed 42 --output data/cluster_4096.txt
./build/nbody \
  --backend cpu \
  --input data/cluster_4096.txt \
  --config configs/cpu_short.cfg \
  --output results/cluster_4096_cpu_short
```

## 输出文件

- `trajectory.bin`：小端 int32 的 P、R，随后是小端 float32 的 `[P,R,3]` 坐标。
- `metadata.json`：记录步号、物理时间、参数和轨迹布局。
- `final_state.csv`：最终位置、速度与质量，保留 double 精度文本。
- `diagnostics.csv`：初态和末态的能量、动量、角动量与质心。
- `performance.json`：力计算时间、模拟时间、粒子步吞吐和守恒误差。

## 下一阶段

以当前 `cuda-naive` 为正确性和性能回归基线，实现共享内存分块的 `cuda-tiled`，随后扫描 block size，并在 4096 粒子、1000 步的相同输入和记录策略下比较 CPU、naive 与 tiled。
