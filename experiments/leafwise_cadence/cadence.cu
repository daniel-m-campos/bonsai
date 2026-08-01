// Stage 0 of the device-leafwise admission plan (docs/architecture/20-cuda-leafwise.md).
// Prices the fixed per-round cost F of the designed leafwise round cadence with
// trivial kernels, so the number measured is launch, sync and staging overhead only.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <queue>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    const cudaError_t status_ = (call);                                      \
    if (status_ != cudaSuccess) {                                            \
      std::fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,       \
                   __LINE__, cudaGetErrorString(status_), #call);            \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

namespace {

constexpr std::int64_t kRootRows = 16'000'000;
constexpr int kLevels = 8;              // 255 splits per tree
constexpr int kSplitsPerTree = 255;
constexpr int kTrees = 100;             // 25,500 rounds, the issue #268 protocol
constexpr int kWarmupRounds = 500;
constexpr std::int64_t kChunkRows = 32'768;
constexpr int kMaxChunks = 64;
constexpr int kFeatures = 100;
constexpr int kBytesPerRow = 12;        // rows uint32 + gh float2

// Every launch touches one element. Stage 0 prices the cadence, not the compute.
__global__ void touch_kernel(float* sink) {
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    sink[0] += 1.0f;
  }
}

struct Config {
  const char* name;
  bool copyback;  // step 3, the D2D range copy-back
  bool syncs;     // steps 4 and 7, the two pinned syncs
  bool floor;     // all grids forced to (1, 1)
};

struct Buffers {
  float* sink = nullptr;
  std::uint8_t* rows_a = nullptr;
  std::uint8_t* rows_b = nullptr;
  std::uint8_t* dev_small = nullptr;  // staging landing zone and D2H sources
  std::uint8_t* host_up = nullptr;    // pinned, 20 B part op + 32 B child sums
  std::uint8_t* host_down = nullptr;  // pinned, 8 B counts + 112 B two FeatBest
};

unsigned chunks_for(std::int64_t rows) {
  const std::int64_t raw = (rows + kChunkRows - 1) / kChunkRows;
  return static_cast<unsigned>(std::clamp<std::int64_t>(raw, 1, kMaxChunks));
}

dim3 grid_of(unsigned x, unsigned y, const Config& cfg) {
  return cfg.floor ? dim3(1, 1) : dim3(x, y);
}

void launch(unsigned gx, unsigned gy, unsigned block, const Config& cfg,
            const Buffers& buf, cudaStream_t stream) {
  touch_kernel<<<grid_of(gx, gy, cfg), block, 0, stream>>>(buf.sink);
  CUDA_CHECK(cudaGetLastError());
}

// One simulated split round on the popped leaf, in the launch order of doc 20.
void round(std::int64_t parent_rows, const Config& cfg, const Buffers& buf,
           cudaStream_t stream, std::priority_queue<double>& heap) {
  const unsigned chunks = chunks_for(parent_rows);
  const unsigned hist_chunks = chunks_for(parent_rows / 2);

  // 1. Stage one PartOpDev for the popped leaf.
  CUDA_CHECK(cudaMemcpyAsync(buf.dev_small, buf.host_up, 20,
                             cudaMemcpyHostToDevice, stream));

  // 2. Partition chain: route, segmented scan, scatter.
  launch(chunks, 1, 256, cfg, buf, stream);
  launch(chunks, 1, 256, cfg, buf, stream);
  launch(chunks, 1, 256, cfg, buf, stream);

  // 3. Copy the parent's range back into the primary row and gradient buffers.
  if (cfg.copyback && !cfg.floor) {
    CUDA_CHECK(cudaMemcpyAsync(buf.rows_a, buf.rows_b,
                               static_cast<std::size_t>(parent_rows) * kBytesPerRow,
                               cudaMemcpyDeviceToDevice, stream));
  }

  // 4. Child counts come down (sync 1).
  CUDA_CHECK(cudaMemcpyAsync(buf.host_down, buf.dev_small, 8,
                             cudaMemcpyDeviceToHost, stream));
  if (cfg.syncs) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // 5. Stage the two children's sums and bounds, build the smaller child,
  //    derive the larger by in-place subtraction.
  CUDA_CHECK(cudaMemcpyAsync(buf.dev_small, buf.host_up, 32,
                             cudaMemcpyHostToDevice, stream));
  launch(hist_chunks, kFeatures, 256, cfg, buf, stream);
  launch(hist_chunks, kFeatures, 256, cfg, buf, stream);
  launch(chunks, 1, 256, cfg, buf, stream);

  // 6. Find over two children x 100 features, then the reduce.
  launch(kFeatures, 2, 128, cfg, buf, stream);
  launch(2, 1, 256, cfg, buf, stream);

  // 7. The two FeatBest come down (sync 2).
  CUDA_CHECK(cudaMemcpyAsync(buf.host_down, buf.dev_small, 112,
                             cudaMemcpyDeviceToHost, stream));
  if (cfg.syncs) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  // 8. Host residue: both children enter the gain heap.
  heap.push(static_cast<double>(parent_rows));
  heap.push(static_cast<double>(parent_rows) * 0.5);
  heap.pop();
  heap.pop();
}

// A balanced 256-leaf tree on 16M rows: level k contributes 2^k splits of
// 16M >> k parent rows, cycled once per tree.
std::vector<std::int64_t> split_schedule() {
  std::vector<std::int64_t> rows;
  rows.reserve(kSplitsPerTree);
  for (int level = 0; level < kLevels; ++level) {
    for (int i = 0; i < (1 << level); ++i) {
      rows.push_back(kRootRows >> level);
    }
  }
  return rows;
}

double run_config(const Config& cfg, const Buffers& buf, cudaStream_t stream,
                  const std::vector<std::int64_t>& schedule) {
  std::priority_queue<double> heap;
  for (int r = 0; r < kWarmupRounds; ++r) {
    round(schedule[r % kSplitsPerTree], cfg, buf, stream, heap);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  const auto start = std::chrono::steady_clock::now();
  for (int tree = 0; tree < kTrees; ++tree) {
    for (int split = 0; split < kSplitsPerTree; ++split) {
      round(schedule[split], cfg, buf, stream, heap);
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  const auto stop = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string pod_id = argc > 1 ? argv[1] : "unknown";

  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  int driver = 0;
  int runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime));

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  Buffers buf;
  const std::size_t row_bytes = static_cast<std::size_t>(kRootRows) * kBytesPerRow;
  CUDA_CHECK(cudaMalloc(&buf.sink, sizeof(float)));
  CUDA_CHECK(cudaMemset(buf.sink, 0, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&buf.rows_a, row_bytes));
  CUDA_CHECK(cudaMalloc(&buf.rows_b, row_bytes));
  CUDA_CHECK(cudaMemset(buf.rows_a, 0, row_bytes));
  CUDA_CHECK(cudaMemset(buf.rows_b, 0, row_bytes));
  CUDA_CHECK(cudaMalloc(&buf.dev_small, 256));
  CUDA_CHECK(cudaMemset(buf.dev_small, 0, 256));
  CUDA_CHECK(cudaMallocHost(&buf.host_up, 256));
  CUDA_CHECK(cudaMallocHost(&buf.host_down, 256));

  const std::vector<std::int64_t> schedule = split_schedule();
  const int rounds = kSplitsPerTree * kTrees;

  const Config configs[4] = {
      {"full", true, true, false},
      {"no_copyback", false, true, false},
      {"no_syncs", false, false, false},
      {"floor", false, true, true},
  };
  double seconds[4] = {0.0, 0.0, 0.0, 0.0};
  double per_round[4] = {0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < 4; ++i) {
    seconds[i] = run_config(configs[i], buf, stream, schedule);
    per_round[i] = seconds[i] * 1e6 / rounds;
  }

  const double fixed_cost = per_round[1];
  const double copyback = per_round[0] - per_round[1];
  const double syncs = per_round[1] - per_round[2];
  const double floor_launch = per_round[3] - syncs;
  const double grid_width = per_round[2] - floor_launch;
  const char* verdict = fixed_cost <= 100.0   ? "under budget"
                        : fixed_cost <= 300.0 ? "over budget, under kill line"
                                              : "over kill line";

  std::printf("{\n");
  std::printf("  \"probe\": \"leafwise-cadence\",\n");
  std::printf("  \"stage\": 0,\n");
  std::printf("  \"doc\": \"docs/architecture/20-cuda-leafwise.md\",\n");
  std::printf("  \"issue\": 268,\n");
  std::printf("  \"pod_id\": \"%s\",\n", pod_id.c_str());
  std::printf("  \"gpu\": \"%s\",\n", prop.name);
  std::printf("  \"sm_count\": %d,\n", prop.multiProcessorCount);
  std::printf("  \"driver_version\": %d,\n", driver);
  std::printf("  \"runtime_version\": %d,\n", runtime);
  std::printf("  \"rounds\": %d,\n", rounds);
  std::printf("  \"trees\": %d,\n", kTrees);
  std::printf("  \"splits_per_tree\": %d,\n", kSplitsPerTree);
  std::printf("  \"root_rows\": %lld,\n", static_cast<long long>(kRootRows));
  std::printf("  \"warmup_rounds\": %d,\n", kWarmupRounds);
  std::printf("  \"configs\": {\n");
  for (int i = 0; i < 4; ++i) {
    std::printf("    \"%s\": {\"seconds\": %.6f, \"us_per_round\": %.3f}%s\n",
                configs[i].name, seconds[i], per_round[i], i == 3 ? "" : ",");
  }
  std::printf("  },\n");
  std::printf("  \"buckets_us\": {\n");
  std::printf("    \"copyback\": %.3f,\n", copyback);
  std::printf("    \"syncs\": %.3f,\n", syncs);
  std::printf("    \"launch_floor\": %.3f,\n", floor_launch);
  std::printf("    \"grid_width\": %.3f\n", grid_width);
  std::printf("  },\n");
  std::printf("  \"fixed_cost_us\": %.3f,\n", fixed_cost);
  std::printf("  \"budget_us\": 100.0,\n");
  std::printf("  \"kill_us\": 300.0,\n");
  std::printf("  \"verdict\": \"%s\"\n", verdict);
  std::printf("}\n");

  CUDA_CHECK(cudaFreeHost(buf.host_down));
  CUDA_CHECK(cudaFreeHost(buf.host_up));
  CUDA_CHECK(cudaFree(buf.dev_small));
  CUDA_CHECK(cudaFree(buf.rows_b));
  CUDA_CHECK(cudaFree(buf.rows_a));
  CUDA_CHECK(cudaFree(buf.sink));
  CUDA_CHECK(cudaStreamDestroy(stream));
  return 0;
}
