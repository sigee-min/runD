#pragma once

#include "../../../../include/rund/compute/abi/primitive.hpp"
#include "../../../../include/rund/compute/fixed.hpp"

#include "collective.hpp"
#include "map.hpp"
#include "primitive.hpp"
#include "storage.hpp"

#include <kernel/program/compute/reduce/model.hpp>
#include <rund/compute/ops.hpp>
#include <rund/compute/stats.hpp>
#include <rund/compute/status.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>

namespace rund::compute::detail {

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
  std::uint64_t trace_dispatch_started_ns{};
  std::uint64_t pending_dispatches{};
  std::uint64_t controlled_count{};
  bool trace_dispatch_active{};
  bool trace_kernel_dispatches{};
  bool controlled_count_valid{};
  std::size_t step{};
  std::size_t reset{};
  CpuPass pass{CpuPass::None};
};

} // namespace rund::compute::detail
