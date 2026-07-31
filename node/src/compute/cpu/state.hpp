#pragma once

#include "../../accel/cpu/scatter/scratch.hpp"
#include "../../accel/cpu/simd/dispatch.hpp"
#include <rund/counter.hpp>

#include <kernel/program/compute/model.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/tile/run.hpp>
#include <rund/compute/abi/model.hpp>
#include <rund/compute/fixed.hpp>
#include <rund/compute/ops.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

namespace rund::compute::detail {

struct BufferState;
struct CpuBufferState;
struct CpuDeviceState;

template <class T> class CpuOverwriteBuffer final {
  static_assert(std::is_trivially_default_constructible_v<T>);
  static_assert(std::is_trivially_destructible_v<T>);

public:
  using value_type = T;

  void allocate(const std::size_t count) {
    values_ =
        count == 0u ? nullptr : std::make_unique_for_overwrite<T[]>(count);
    size_ = count;
  }

  [[nodiscard]] T *data() noexcept { return values_.get(); }
  [[nodiscard]] const T *data() const noexcept { return values_.get(); }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t capacity() const noexcept { return size_; }

private:
  std::unique_ptr<T[]> values_;
  std::size_t size_{};
};

struct alignas(64) CpuSimdCount final {
  std::uint64_t vectors{};
  std::uint64_t tails{};
};

struct CpuProgram final {
  kernel::ComputeMap map;
  node::accel::cpu_simd_detail::CpuSimdDispatch dispatch;
  std::vector<std::uint64_t> input_bytes;
  std::vector<kernel::ReadRoute> read_routes;
  kernel::ComputeTileExecutor tile_plan;
  std::size_t scratch_words{};
  std::uint32_t workers{};
  std::uint64_t tile_size{};
};

struct CpuMapRun;

struct CpuMapTileContext final {
  CpuProgram *program = nullptr;
  CpuMapRun *run = nullptr;
  const std::atomic_bool *cancel = nullptr;
};

struct CpuMapRun final {
  kernel::ComputeTileExecutor tiles;
  std::vector<CpuOverwriteBuffer<std::max_align_t>> scratch;
  std::vector<CpuSimdCount> simd;
  std::array<node::accel::cpu_simd_detail::CpuSimdReadBinding,
             kernel::kMaxComputeBindingCount>
      reads{};
  std::array<node::accel::cpu_simd_detail::CpuSimdWriteBinding, MaxOutputs>
      writes{};
  node::accel::cpu_simd_detail::CpuSimdBindingView bindings{};
  CpuMapTileContext tile{};
  // Buffer owners, addresses, strides, and port counts are frozen until an
  // atomic resident write publishes a different input-owner set.
  bool bindings_frozen{};
};

inline void reset_simd(CpuMapRun &run) noexcept {
  for (CpuSimdCount &count : run.simd) {
    count = {};
  }
}

inline void record_simd(CpuMapRun &run, const std::uint32_t worker,
                        const node::accel::CpuSimdRunResult &result) noexcept {
  if (worker >= run.simd.size()) {
    return;
  }
  ::rund::detail::counter::Accumulate(run.simd[worker].vectors,
                                      result.vector_chunk_count);
  ::rund::detail::counter::Accumulate(run.simd[worker].tails,
                                      result.tail_chunk_count);
}

[[nodiscard]] inline CpuSimdCount sum_simd(const CpuMapRun &run) noexcept {
  CpuSimdCount total{};
  for (const CpuSimdCount &count : run.simd) {
    ::rund::detail::counter::Accumulate(total.vectors, count.vectors);
    ::rund::detail::counter::Accumulate(total.tails, count.tails);
  }
  return total;
}

struct CpuRuntimeGraph;

struct ResetRoute final {
  std::uint32_t value_index{};
  std::uint32_t step{};
  std::uint32_t last{};
};

struct CpuCollective final {
  kernel::ComputeTileExecutor tile_plan;
  std::uint64_t tile_size{};
  std::uint32_t tile_count{};
  bool needs_prefixes{};
};

// A collective is bounded by kernel::u32 elements. The largest magnitude is
// therefore below 2^96 for every supported 32/64-bit integer or fixed lane,
// so a signed 128-bit carrier preserves the mathematical sum until the one
// observable storage boundary.
using CpuCollectiveWide = __int128_t;

struct CpuCollectiveRun final {
  kernel::ComputeTileExecutor tiles;
  std::vector<CpuCollectiveWide> totals;
  std::vector<CpuCollectiveWide> prefixes;
  std::uint64_t tile_size{};
  bool needs_prefixes{};
};

template <class Key> struct CpuSortPrimitiveScratch final {
  CpuOverwriteBuffer<Key> keys;
  CpuOverwriteBuffer<kernel::u32> values;
  std::array<kernel::u64, 256u> counts{};
  std::array<kernel::u64, 256u> offsets{};
};

struct CpuScatterPrimitiveScratch final {
  node::accel::detail::CpuScatterScratch values;
};

struct CpuScatterReducePrimitiveScratch final {
  CpuOverwriteBuffer<kernel::u32> sorted_indices;
};

template <class Lane> struct CpuTransformScratch final {
  CpuOverwriteBuffer<Lane> twiddle;
};

template <class Lane> struct CpuFactorQrScratch final {
  std::array<CpuOverwriteBuffer<Lane>, 3u> values;
};

template <class Lane> struct CpuSolveLuScratch final {
  CpuOverwriteBuffer<Lane> factor;
  CpuOverwriteBuffer<kernel::u32> pivots;
};

template <class Lane> struct CpuSolveCholeskyScratch final {
  CpuOverwriteBuffer<Lane> factor;
};

template <class Lane> struct CpuSolveQrMatrixScratch final {
  CpuOverwriteBuffer<Lane> y;
  std::array<CpuOverwriteBuffer<Lane>, 3u> qr;
};

template <class Lane> struct CpuSolveQrFactorScratch final {
  CpuOverwriteBuffer<Lane> y;
};

template <class Lane> struct CpuSpectrumEigenScratch final {
  std::array<CpuOverwriteBuffer<Lane>, 3u> values;
};

template <class Lane> struct CpuSpectrumSvdValuesScratch final {
  std::array<CpuOverwriteBuffer<Lane>, 3u> values;
  CpuOverwriteBuffer<kernel::u64> order;
};

template <class Lane> struct CpuSpectrumSvdVectorsScratch final {
  std::array<CpuOverwriteBuffer<Lane>, 4u> values;
  CpuOverwriteBuffer<kernel::u64> order;
};

using CpuPrimitiveScratch =
    std::variant<std::monostate,
                 std::unique_ptr<CpuSortPrimitiveScratch<kernel::u32>>,
                 std::unique_ptr<CpuSortPrimitiveScratch<kernel::u64>>,
                 std::unique_ptr<CpuScatterPrimitiveScratch>,
                 std::unique_ptr<CpuScatterReducePrimitiveScratch>,
                 std::unique_ptr<CpuTransformScratch<kernel::i32>>,
                 std::unique_ptr<CpuTransformScratch<kernel::i64>>,
                 std::unique_ptr<CpuFactorQrScratch<kernel::i32>>,
                 std::unique_ptr<CpuFactorQrScratch<kernel::i64>>,
                 std::unique_ptr<CpuSolveLuScratch<kernel::i32>>,
                 std::unique_ptr<CpuSolveLuScratch<kernel::i64>>,
                 std::unique_ptr<CpuSolveCholeskyScratch<kernel::i32>>,
                 std::unique_ptr<CpuSolveCholeskyScratch<kernel::i64>>,
                 std::unique_ptr<CpuSolveQrMatrixScratch<kernel::i32>>,
                 std::unique_ptr<CpuSolveQrMatrixScratch<kernel::i64>>,
                 std::unique_ptr<CpuSolveQrFactorScratch<kernel::i32>>,
                 std::unique_ptr<CpuSolveQrFactorScratch<kernel::i64>>,
                 std::unique_ptr<CpuSpectrumEigenScratch<kernel::i32>>,
                 std::unique_ptr<CpuSpectrumEigenScratch<kernel::i64>>,
                 std::unique_ptr<CpuSpectrumSvdValuesScratch<kernel::i32>>,
                 std::unique_ptr<CpuSpectrumSvdValuesScratch<kernel::i64>>,
                 std::unique_ptr<CpuSpectrumSvdVectorsScratch<kernel::i32>>,
                 std::unique_ptr<CpuSpectrumSvdVectorsScratch<kernel::i64>>>;

struct CpuGraphProgram final {
  ~CpuGraphProgram();

  std::unique_ptr<CpuRuntimeGraph> runtime;
  std::vector<std::unique_ptr<CpuProgram>> maps;
  std::vector<std::unique_ptr<CpuCollective>> collectives;
  std::vector<std::size_t> bind_begin;
  std::vector<std::size_t> bind_count;
  std::vector<ResetRoute> resets;
  std::uint64_t graph_hash{};
};

struct CpuGraphRun final {
  std::vector<std::unique_ptr<CpuMapRun>> maps;
  std::vector<std::unique_ptr<CpuCollectiveRun>> collectives;
  std::vector<CpuPrimitiveScratch> scratch;
  CpuPrimitiveScratch empty_scratch{};
  std::vector<std::shared_ptr<BufferState>> owned_buffers;
  std::span<const std::shared_ptr<BufferState>> buffers;
  const std::shared_ptr<BufferState> *bound_inputs = nullptr;
  Primitive semantic_primitive{Primitive::Reduce};
  std::uint32_t semantic_status{};
  std::uint64_t semantic_failure_count{};
  std::uint64_t conflict_count{};
  std::uint64_t overflow_ordinal{ControlStats::no_overflow};
};

enum class CpuPass : std::uint8_t {
  None,
  Map,
  ScanLocal,
  ScanCorrect,
  ReduceLocal,
  Primitive,
};

enum class CpuCollectiveKind : std::uint8_t {
  Scan,
  Reduce,
};

struct CpuCollectiveTileContext final {
  CpuCollectiveRun *run = nullptr;
  const void *input = nullptr;
  void *output = nullptr;
  CpuCollectiveKind kind{CpuCollectiveKind::Scan};
  CpuPass pass{CpuPass::None};
  Scan scan{Scan::InclusiveSum};
  kernel::ReduceOp reduce{kernel::ReduceOp::Sum};
  Type type{Type::U32};
  const std::atomic_bool *cancel = nullptr;
};

struct CpuRun final {
  std::unique_ptr<CpuGraphRun> graph;
  CpuCollectiveTileContext tile{};
  kernel::Partition primitive_partition{};
  kernel::WorkerSubmission primitive_submission{};
  Status primitive_status{Status::fail(Reason::PrimitiveNotReady)};
  void *primitive_ready_context = nullptr;
  void (*primitive_ready)(void *) noexcept = nullptr;
  Stats stats{};
  std::uint64_t pending_dispatches{};
  std::uint64_t controlled_count{};
  bool controlled_count_valid{};
  std::size_t step{};
  std::size_t reset{};
  CpuPass pass{CpuPass::None};
};

} // namespace rund::compute::detail
