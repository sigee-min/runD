#pragma once

#include "arena.hpp"

#include "../../accel/cpu/scatter/scratch.hpp"
#include "../../accel/cpu/simd/dispatch.hpp"
#include "../../accel/kernel/view.hpp"
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
#include <optional>
#include <span>
#include <type_traits>
#include <variant>
#include <vector>

namespace rund::compute::detail {

struct BufferState;
struct CpuViewTransfer;
struct JobBufferView;
struct JobWorkspace;
struct CpuBufferState;
struct CpuDeviceState;
class CpuPreparedArena;

// Value-lifetime owner with pointer ergonomics. It is used only when the
// pointee cannot outlive its containing state; unlike unique_ptr it creates no
// allocator or control-block authority.
template <class T> class EmbeddedOwner final {
public:
  EmbeddedOwner() noexcept = default;
  EmbeddedOwner(const EmbeddedOwner &) = delete;
  EmbeddedOwner &operator=(const EmbeddedOwner &) = delete;
  EmbeddedOwner(EmbeddedOwner &&) noexcept = default;
  EmbeddedOwner &operator=(EmbeddedOwner &&) noexcept = default;

  template <class... Args> T &emplace(Args &&...args) {
    return value_.emplace(std::forward<Args>(args)...);
  }
  void reset() noexcept { value_.reset(); }

  [[nodiscard]] T *get() noexcept {
    return value_.has_value() ? std::addressof(*value_) : nullptr;
  }
  [[nodiscard]] const T *get() const noexcept {
    return value_.has_value() ? std::addressof(*value_) : nullptr;
  }
  [[nodiscard]] T *operator->() noexcept { return get(); }
  [[nodiscard]] const T *operator->() const noexcept { return get(); }
  [[nodiscard]] T &operator*() noexcept { return *value_; }
  [[nodiscard]] const T &operator*() const noexcept { return *value_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return value_.has_value();
  }
  [[nodiscard]] bool has_value() const noexcept { return value_.has_value(); }

  [[nodiscard]] friend bool operator==(const EmbeddedOwner &owner,
                                       std::nullptr_t) noexcept {
    return !owner.value_.has_value();
  }
  [[nodiscard]] friend bool operator!=(const EmbeddedOwner &owner,
                                       std::nullptr_t) noexcept {
    return owner.value_.has_value();
  }

private:
  std::optional<T> value_{};
};

struct alignas(64) CpuSimdCount final {
  std::uint64_t vectors{};
  std::uint64_t tails{};
};

struct CpuProgram final {
  kernel::ComputeMap map;
  node::accel::cpu_simd_detail::CpuSimdDispatch dispatch;
  std::vector<std::uint64_t> input_bytes;
  std::vector<std::uint64_t> input_counts;
  std::vector<kernel::ReadRoute> read_routes;
  kernel::ComputeTileExecutor tile_plan;
  std::size_t scratch_words{};
  std::uint32_t workers{};
  std::uint64_t tile_size{};
};

struct CpuMapRun;
struct CpuMapRoute;

struct CpuMapTileContext final {
  CpuProgram *program = nullptr;
  CpuMapRun *run = nullptr;
  CpuMapRoute *route = nullptr;
  const std::atomic_bool *cancel = nullptr;
};

struct CpuMapRun final {
  kernel::ComputeTileRunPlan tile_plan;
  kernel::ComputeTileExecutor tiles;
  // Pipeline/Program steps are serial. These typed views borrow the active
  // execution arena's component-wise maximum instead of owning one allocation
  // per Map and worker.
  std::span<std::max_align_t> scratch;
  std::span<CpuSimdCount> simd;
  CpuPreparedArena *execution{};
  std::size_t scratch_words{};
  std::size_t workers{};
};

// One immutable Pipeline route owns only resolved Buffer addresses and Views.
// Tile scheduling, worker scratch, and SIMD counters are mutable execution
// storage shared by every serial route of the same Program.
struct CpuMapRoute final {
  node::accel::cpu_simd_detail::CpuSimdBindingView bindings{};
  CpuMapTileContext tile{};
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
  kernel::ComputeTileRunPlan tile_plan;
  kernel::ComputeTileExecutor tiles;
  std::span<CpuCollectiveWide> total_capacity;
  std::span<CpuCollectiveWide> prefix_capacity;
  std::span<CpuCollectiveWide> totals;
  std::span<CpuCollectiveWide> prefixes;
  CpuPreparedArena *execution{};
  std::uint64_t tile_size{};
  bool needs_prefixes{};
};

// One mutable typed max-envelope serves every serial Map/collective in a
// standalone Program or an entire Pipeline. The immutable per-step plans stay
// with CpuMapRun/CpuCollectiveRun; only this storage is rebound between steps.
struct CpuExecutionStoragePlan final {
  kernel::ComputeTileRunStoragePlan tiles{};
  std::size_t map_scratch_count{};
  std::size_t simd_count{};
  std::size_t collective_total_count{};
  std::size_t collective_prefix_count{};
  std::size_t primitive_u32_count{};
  std::size_t primitive_u64_count{};
  std::size_t primitive_i32_count{};
  std::size_t primitive_i64_count{};
  std::size_t scatter_slot_count{};
  std::size_t transform_i32_count{};
  std::size_t transform_i64_count{};
  // Scratch descriptors are persistent per primitive and therefore additive;
  // their payload buffers remain the component-wise maximum above. Storage is
  // max_align_t padded for placement while payload preserves exact accounting.
  std::size_t primitive_object_storage_bytes{};
  std::size_t primitive_object_payload_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const CpuExecutionStoragePlan &other) const noexcept {
    const auto &left_memory = tiles.retained.memory;
    const auto &right_memory = other.tiles.retained.memory;
    return tiles.failure_slot_capacity == other.tiles.failure_slot_capacity &&
           tiles.worker_capacity == other.tiles.worker_capacity &&
           tiles.ok == other.tiles.ok &&
           tiles.retained.ok == other.tiles.retained.ok &&
           left_memory.state_bytes == right_memory.state_bytes &&
           left_memory.workspace_bytes == right_memory.workspace_bytes &&
           left_memory.failure_slot_bytes == right_memory.failure_slot_bytes &&
           left_memory.worker_tile_bytes == right_memory.worker_tile_bytes &&
           left_memory.async_context_bytes ==
               right_memory.async_context_bytes &&
           left_memory.total_bytes == right_memory.total_bytes &&
           map_scratch_count == other.map_scratch_count &&
           simd_count == other.simd_count &&
           collective_total_count == other.collective_total_count &&
           collective_prefix_count == other.collective_prefix_count &&
           primitive_u32_count == other.primitive_u32_count &&
           primitive_u64_count == other.primitive_u64_count &&
           primitive_i32_count == other.primitive_i32_count &&
           primitive_i64_count == other.primitive_i64_count &&
           scatter_slot_count == other.scatter_slot_count &&
           transform_i32_count == other.transform_i32_count &&
           transform_i64_count == other.transform_i64_count &&
           primitive_object_storage_bytes ==
               other.primitive_object_storage_bytes &&
           primitive_object_payload_bytes ==
               other.primitive_object_payload_bytes;
  }
};

template <class Key> struct CpuSortPrimitiveScratch final {
  std::span<Key> keys;
  std::span<kernel::u32> values;
  std::array<kernel::u64, 256u> counts{};
  std::array<kernel::u64, 256u> offsets{};
};

struct CpuScatterPrimitiveScratch final {
  std::span<kernel::u32> keys;
  std::span<kernel::u32> marks;
  // All Scatter descriptors in one serial execution envelope share these
  // physical marks and their epoch. mark_capacity covers the complete max
  // slab so epoch wrap clears stale marks outside a smaller active prefix.
  std::span<kernel::u32> mark_capacity;
  kernel::u32 *epoch{};

  [[nodiscard]] bool scatter_ready() const noexcept { return epoch != nullptr; }
  [[nodiscard]] kernel::u32 &scatter_epoch() noexcept { return *epoch; }
  [[nodiscard]] std::span<kernel::u32> scatter_marks() noexcept {
    return mark_capacity;
  }
};

struct CpuScatterReducePrimitiveScratch final {
  std::span<kernel::u32> sorted_indices;
};

template <class Lane> struct CpuTransformScratch final {
  std::span<Lane> twiddle;
};

template <class Lane> struct CpuFactorQrScratch final {
  std::array<std::span<Lane>, 3u> values;
};

template <class Lane> struct CpuSolveLuScratch final {
  std::span<Lane> factor;
  std::span<kernel::u32> pivots;
};

template <class Lane> struct CpuSolveCholeskyScratch final {
  std::span<Lane> factor;
};

template <class Lane> struct CpuSolveQrMatrixScratch final {
  std::span<Lane> y;
  std::array<std::span<Lane>, 3u> qr;
};

template <class Lane> struct CpuSolveQrFactorScratch final {
  std::span<Lane> y;
};

template <class Lane> struct CpuSpectrumEigenScratch final {
  std::array<std::span<Lane>, 3u> values;
};

template <class Lane> struct CpuSpectrumSvdValuesScratch final {
  std::array<std::span<Lane>, 3u> values;
  std::span<kernel::u64> order;
};

template <class Lane> struct CpuSpectrumSvdVectorsScratch final {
  std::array<std::span<Lane>, 4u> values;
  std::span<kernel::u64> order;
};

using CpuPrimitiveScratch = std::variant<
    std::monostate, CpuSortPrimitiveScratch<kernel::u32> *,
    CpuSortPrimitiveScratch<kernel::u64> *, CpuScatterPrimitiveScratch *,
    CpuScatterReducePrimitiveScratch *, CpuTransformScratch<kernel::i32> *,
    CpuTransformScratch<kernel::i64> *, CpuFactorQrScratch<kernel::i32> *,
    CpuFactorQrScratch<kernel::i64> *, CpuSolveLuScratch<kernel::i32> *,
    CpuSolveLuScratch<kernel::i64> *, CpuSolveCholeskyScratch<kernel::i32> *,
    CpuSolveCholeskyScratch<kernel::i64> *,
    CpuSolveQrMatrixScratch<kernel::i32> *,
    CpuSolveQrMatrixScratch<kernel::i64> *,
    CpuSolveQrFactorScratch<kernel::i32> *,
    CpuSolveQrFactorScratch<kernel::i64> *,
    CpuSpectrumEigenScratch<kernel::i32> *,
    CpuSpectrumEigenScratch<kernel::i64> *,
    CpuSpectrumSvdValuesScratch<kernel::i32> *,
    CpuSpectrumSvdValuesScratch<kernel::i64> *,
    CpuSpectrumSvdVectorsScratch<kernel::i32> *,
    CpuSpectrumSvdVectorsScratch<kernel::i64> *>;

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

// Mutable Map/collective/primitive execution storage. A standalone Job owns a
// private instance. A Pipeline owns one instance per Program and shares it
// only across routes proved serial by the Pipeline gate.
struct CpuGraphStorage final {
  const CpuGraphProgram *program = nullptr;
  // Lifetime edge to the one arena whose execution spans are bound below.
  // Route and Job binding slices share this same physical authority.
  std::shared_ptr<CpuPreparedArena> prepared_arena;
  // Program steps address compact dense run arrays through immutable indices.
  // Exact reserve before emplacement removes one heap owner and one pointer
  // chase per Map/collective while preserving O(1) step lookup.
  std::vector<CpuMapRun> maps;
  std::vector<CpuCollectiveRun> collectives;
  std::vector<std::size_t> map_by_step;
  std::vector<std::size_t> collective_by_step;
  std::vector<CpuPrimitiveScratch> scratch;
  CpuPrimitiveScratch empty_scratch{};
};

inline constexpr std::size_t NoCpuGraphStorageIndex = ~std::size_t{0};

[[nodiscard]] inline CpuMapRun *cpu_map_run(CpuGraphStorage &storage,
                                            const std::size_t step) noexcept {
  if (step >= storage.map_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.map_by_step[step];
  return index < storage.maps.size() ? &storage.maps[index] : nullptr;
}

[[nodiscard]] inline const CpuMapRun *
cpu_map_run(const CpuGraphStorage &storage, const std::size_t step) noexcept {
  if (step >= storage.map_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.map_by_step[step];
  return index < storage.maps.size() ? &storage.maps[index] : nullptr;
}

[[nodiscard]] inline CpuCollectiveRun *
cpu_collective_run(CpuGraphStorage &storage, const std::size_t step) noexcept {
  if (step >= storage.collective_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.collective_by_step[step];
  return index < storage.collectives.size() ? &storage.collectives[index]
                                            : nullptr;
}

[[nodiscard]] inline const CpuCollectiveRun *
cpu_collective_run(const CpuGraphStorage &storage,
                   const std::size_t step) noexcept {
  if (step >= storage.collective_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = storage.collective_by_step[step];
  return index < storage.collectives.size() ? &storage.collectives[index]
                                            : nullptr;
}

struct CpuStorageBytes final {
  std::uint64_t host{};
  std::uint64_t tile{};

  [[nodiscard]] constexpr bool
  operator==(const CpuStorageBytes &) const noexcept = default;
};

// Allocation-free sealed view over one immutable CpuGraphProgram. Per-step
// plans stay with that Program; this descriptor freezes their checked retained
// extent and the exact top-level allocation counts without copying a second
// plan table.
struct CpuGraphStoragePlan final {
  const CpuGraphProgram *program = nullptr;
  const CpuRuntimeGraph *runtime = nullptr;
  std::uint64_t graph_hash{};
  std::size_t step_count{};
  std::size_t map_count{};
  std::size_t collective_count{};
  std::size_t scratch_count{};
  std::size_t scratch_slots{};
  CpuStorageBytes containers{};
  CpuStorageBytes maps{};
  CpuStorageBytes collectives{};
  // Per-Program retained state excludes the Pipeline/Job-owned prepared arena.
  CpuStorageBytes private_total{};
  CpuExecutionStoragePlan execution{};

  [[nodiscard]] constexpr bool
  operator==(const CpuGraphStoragePlan &) const noexcept = default;
};

struct CpuRunRoutePlan final {
  const CpuGraphProgram *program = nullptr;
  const CpuRuntimeGraph *runtime = nullptr;
  std::uint64_t graph_hash{};
  std::size_t step_count{};
  std::size_t map_count{};
  std::size_t read_count{};
  std::size_t write_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuRunRoutePlan &) const noexcept = default;
};

struct CpuRunRouteSlice final {
  std::size_t map_begin{};
  std::size_t map_count{};
  std::size_t read_begin{};
  std::size_t read_count{};
  std::size_t write_begin{};
  std::size_t write_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuRunRouteSlice &) const noexcept = default;
};

// Exact immutable slice for one private Pipeline Job. Buffer owners and Views
// are mutable only during cold preparation (dense-view staging may rewrite a
// binding); warm execution reads the frozen prefix without allocating.
struct CpuJobBindingSlice final {
  std::size_t input_begin{};
  std::size_t input_count{};
  std::size_t output_begin{};
  std::size_t output_count{};
  std::size_t input_view_begin{};
  std::size_t input_view_count{};
  std::size_t output_view_begin{};
  std::size_t output_view_count{};
  std::size_t kernel_view_begin{};
  std::size_t kernel_view_count{};
  std::size_t input_transfer_begin{};
  std::size_t input_transfer_count{};
  std::size_t output_transfer_begin{};
  std::size_t output_transfer_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuJobBindingSlice &) const noexcept = default;
};

struct CpuJobBindingCounts final {
  std::size_t inputs{};
  std::size_t outputs{};
  std::size_t kernel_views{};
  std::size_t input_transfers{};
  std::size_t output_transfers{};

  [[nodiscard]] constexpr bool
  operator==(const CpuJobBindingCounts &) const noexcept = default;
};

struct CpuJobBindingStorage final {
  std::span<std::shared_ptr<BufferState>> inputs{};
  std::span<std::shared_ptr<BufferState>> outputs{};
  std::span<JobBufferView> input_views{};
  std::span<JobBufferView> output_views{};
  std::span<node::accel::detail::KernelViewSlot> kernel_views{};
  std::span<CpuViewTransfer> input_transfers{};
  std::span<CpuViewTransfer> output_transfers{};
};

struct CpuWorkspaceSlice final {
  std::size_t workspace_begin{};
  std::size_t workspace_count{};
  std::size_t buffer_begin{};
  std::size_t buffer_count{};
  std::size_t offset_begin{};
  std::size_t offset_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuWorkspaceSlice &) const noexcept = default;
};

struct CpuWorkspaceStorage final {
  JobWorkspace *workspace{};
  std::span<std::shared_ptr<BufferState>> buffers{};
  std::span<std::size_t> offsets{};
};

struct CpuPreparedArenaPlan final {
  CpuArenaLayout layout{};
  CpuExecutionStoragePlan execution{};
  CpuArenaSegment tile_state{};
  CpuArenaSegment failure_slots{};
  CpuArenaSegment worker_tiles{};
  CpuArenaSegment worker_stats_partitions{};
  CpuArenaSegment worker_stats_start_offset_ns{};
  CpuArenaSegment worker_stats_elapsed_ns{};
  CpuArenaSegment worker_stats_tail_wait_ns{};
  CpuArenaSegment map_scratch{};
  CpuArenaSegment simd{};
  CpuArenaSegment collective_totals{};
  CpuArenaSegment collective_prefixes{};
  CpuArenaSegment primitive_u32{};
  CpuArenaSegment primitive_u64{};
  CpuArenaSegment primitive_i32{};
  CpuArenaSegment primitive_i64{};
  CpuArenaSegment scatter_keys{};
  CpuArenaSegment scatter_marks{};
  CpuArenaSegment scatter_epoch{};
  CpuArenaSegment transform_i32{};
  CpuArenaSegment transform_i64{};
  CpuArenaSegment primitive_objects{};
  CpuArenaSegment maps{};
  CpuArenaSegment reads{};
  CpuArenaSegment writes{};
  CpuArenaSegment buffer_owners{};
  CpuArenaSegment buffer_views{};
  CpuArenaSegment kernel_views{};
  CpuArenaSegment view_transfers{};
  CpuArenaSegment workspaces{};
  CpuArenaSegment workspace_offsets{};
  std::size_t map_count{};
  std::size_t read_count{};
  std::size_t write_count{};
  std::size_t buffer_owner_count{};
  std::size_t buffer_view_count{};
  std::size_t kernel_view_count{};
  std::size_t view_transfer_count{};
  std::size_t workspace_count{};
  std::size_t workspace_offset_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuPreparedArenaPlan &) const noexcept = default;
};

class CpuPreparedArena final {
public:
  CpuPreparedArena() noexcept = default;
  CpuPreparedArena(const CpuPreparedArena &) = delete;
  CpuPreparedArena &operator=(const CpuPreparedArena &) = delete;
  CpuPreparedArena(CpuPreparedArena &&) = delete;
  CpuPreparedArena &operator=(CpuPreparedArena &&) = delete;
  ~CpuPreparedArena();

  [[nodiscard]] bool
  supports(const CpuExecutionStoragePlan &plan) const noexcept;
  [[nodiscard]] kernel::ComputeTileRunStorageView tile_storage() noexcept;
  [[nodiscard]] std::span<std::max_align_t> map_scratch() noexcept {
    return map_scratch_;
  }
  [[nodiscard]] std::span<CpuSimdCount> simd() noexcept { return simd_; }
  [[nodiscard]] std::span<CpuCollectiveWide> collective_totals() noexcept {
    return collective_totals_;
  }
  [[nodiscard]] std::span<CpuCollectiveWide> collective_prefixes() noexcept {
    return collective_prefixes_;
  }
  [[nodiscard]] std::span<kernel::u32> primitive_u32() noexcept {
    return primitive_u32_;
  }
  [[nodiscard]] std::span<kernel::u64> primitive_u64() noexcept {
    return primitive_u64_;
  }
  [[nodiscard]] std::span<kernel::i32> primitive_i32() noexcept {
    return primitive_i32_;
  }
  [[nodiscard]] std::span<kernel::i64> primitive_i64() noexcept {
    return primitive_i64_;
  }
  [[nodiscard]] std::span<kernel::u32> scatter_keys() noexcept {
    return scatter_keys_;
  }
  [[nodiscard]] std::span<kernel::u32> scatter_marks() noexcept {
    return scatter_marks_;
  }
  [[nodiscard]] kernel::u32 *scatter_epoch() noexcept {
    return scatter_epoch_.size() == 1u ? scatter_epoch_.data() : nullptr;
  }
  [[nodiscard]] std::span<kernel::i32>
  claim_transform_i32(std::size_t count) noexcept;
  [[nodiscard]] std::span<kernel::i64>
  claim_transform_i64(std::size_t count) noexcept;
  template <class Scratch>
  [[nodiscard]] Scratch *claim_primitive_object() noexcept {
    static_assert(std::is_nothrow_default_constructible_v<Scratch>);
    static_assert(std::is_trivially_destructible_v<Scratch>);
    static_assert(alignof(Scratch) <= alignof(std::max_align_t));
    constexpr std::size_t alignment = alignof(std::max_align_t);
    constexpr std::size_t storage_bytes =
        (sizeof(Scratch) + alignment - 1u) / alignment * alignment;
    if (primitive_objects_claimed_ > primitive_objects_.size() ||
        storage_bytes >
            primitive_objects_.size() - primitive_objects_claimed_) {
      return nullptr;
    }
    void *const address =
        primitive_objects_.data() + primitive_objects_claimed_;
    primitive_objects_claimed_ += storage_bytes;
    return std::construct_at(static_cast<Scratch *>(address));
  }
  [[nodiscard]] bool
  claims_complete(const CpuExecutionStoragePlan &plan) const noexcept {
    return transform_i32_claimed_ == plan.transform_i32_count &&
           transform_i64_claimed_ == plan.transform_i64_count &&
           primitive_objects_claimed_ == plan.primitive_object_storage_bytes;
  }
  [[nodiscard]] bool
  view(const CpuRunRouteSlice &slice, std::span<CpuMapRoute> &maps,
       std::span<node::accel::cpu_simd_detail::CpuSimdReadBinding> &reads,
       std::span<node::accel::cpu_simd_detail::CpuSimdWriteBinding>
           &writes) noexcept;
  [[nodiscard]] bool view(const CpuJobBindingSlice &slice,
                          CpuJobBindingStorage &storage) noexcept;
  [[nodiscard]] bool view(const CpuWorkspaceSlice &slice,
                          CpuWorkspaceStorage &storage) noexcept;
  [[nodiscard]] std::uint64_t extent_bytes() const noexcept {
    return mapping_.extent_bytes();
  }
  [[nodiscard]] std::uint64_t committed_bytes() const noexcept {
    return mapping_.committed_bytes();
  }
  [[nodiscard]] std::uint64_t payload_host_bytes() const noexcept {
    return payload_host_bytes_;
  }
  [[nodiscard]] std::uint64_t payload_tile_bytes() const noexcept {
    return payload_tile_bytes_;
  }

private:
  friend Result<std::shared_ptr<CpuPreparedArena>>
  make_cpu_prepared_arena(const CpuPreparedArenaPlan &plan);

  [[nodiscard]] bool materialize(const CpuPreparedArenaPlan &plan) noexcept;
  void clear() noexcept;
  CpuArenaMapping mapping_{};
  kernel::ComputeTileRunStoragePlan tile_capacity_{};
  std::span<kernel::ComputeTileRunStorage> tile_state_{};
  std::span<const char *> failure_slots_{};
  std::span<kernel::u32> worker_tiles_{};
  std::span<kernel::u32> worker_stats_partitions_{};
  std::span<kernel::u64> worker_stats_start_offset_ns_{};
  std::span<kernel::u64> worker_stats_elapsed_ns_{};
  std::span<kernel::u64> worker_stats_tail_wait_ns_{};
  std::span<std::max_align_t> map_scratch_{};
  std::span<CpuSimdCount> simd_{};
  std::span<CpuCollectiveWide> collective_totals_{};
  std::span<CpuCollectiveWide> collective_prefixes_{};
  std::span<kernel::u32> primitive_u32_{};
  std::span<kernel::u64> primitive_u64_{};
  std::span<kernel::i32> primitive_i32_{};
  std::span<kernel::i64> primitive_i64_{};
  std::span<kernel::u32> scatter_keys_{};
  std::span<kernel::u32> scatter_marks_{};
  std::span<kernel::u32> scatter_epoch_{};
  std::span<kernel::i32> transform_i32_{};
  std::span<kernel::i64> transform_i64_{};
  std::span<std::byte> primitive_objects_{};
  std::span<CpuMapRoute> maps_{};
  std::span<node::accel::cpu_simd_detail::CpuSimdReadBinding> reads_{};
  std::span<node::accel::cpu_simd_detail::CpuSimdWriteBinding> writes_{};
  std::span<std::shared_ptr<BufferState>> buffer_owners_{};
  std::span<JobBufferView> buffer_views_{};
  std::span<node::accel::detail::KernelViewSlot> kernel_views_{};
  std::span<CpuViewTransfer> view_transfers_{};
  std::span<JobWorkspace> workspaces_{};
  std::span<std::size_t> workspace_offsets_{};
  std::size_t transform_i32_claimed_{};
  std::size_t transform_i64_claimed_{};
  std::size_t primitive_objects_claimed_{};
  std::uint64_t payload_host_bytes_{};
  std::uint64_t payload_tile_bytes_{};
};

// Route state is deliberately separate from CpuGraphStorage: occurrence-
// specific Buffer/View identity must stay frozen while execution scratch is
// reused. The flat binding arrays retain exactly the active port count rather
// than kMaxComputeBindingCount entries for every Map.
struct CpuGraphRun final {
  std::shared_ptr<CpuGraphStorage> storage;
  std::span<CpuMapRoute> maps;
  std::span<node::accel::cpu_simd_detail::CpuSimdReadBinding> reads;
  std::span<node::accel::cpu_simd_detail::CpuSimdWriteBinding> writes;
  CpuPrimitiveScratch empty_scratch{};
  std::span<const std::shared_ptr<BufferState>> buffers;
  const std::shared_ptr<BufferState> *bound_inputs = nullptr;
  Primitive semantic_primitive{Primitive::Reduce};
  std::uint32_t semantic_status{};
  std::uint64_t semantic_failure_count{};
  std::uint64_t conflict_count{};
  std::uint64_t overflow_ordinal{ControlStats::no_overflow};
};

[[nodiscard]] inline CpuMapRoute *
cpu_map_route(CpuGraphRun &run, const std::size_t step) noexcept {
  if (run.storage == nullptr || step >= run.storage->map_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = run.storage->map_by_step[step];
  return index < run.maps.size() ? &run.maps[index] : nullptr;
}

[[nodiscard]] inline const CpuMapRoute *
cpu_map_route(const CpuGraphRun &run, const std::size_t step) noexcept {
  if (run.storage == nullptr || step >= run.storage->map_by_step.size()) {
    return nullptr;
  }
  const std::size_t index = run.storage->map_by_step[step];
  return index < run.maps.size() ? &run.maps[index] : nullptr;
}

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
  EmbeddedOwner<CpuGraphRun> graph;
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
