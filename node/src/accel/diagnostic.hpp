#pragma once

#include <accel/check.hpp>
#include <accel/context/value.hpp>
#include <accel/kernel/run.hpp>
#include <accel/kernel/value.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace rund::node::accel::diagnostic {

enum class Phase : std::uint8_t {
  KernelCheckProjection,
  ScratchExecutionAdmission,
  ScratchPlanProjection,
  PrepareRunAdmission,
  PrepareRunShape,
  PrepareRunBinds,
  PrepareRunResets,
  PrepareRunDispatch,
  PrepareRunOrder,
  PrepareRunBackend,
  PrepareRunProjection,
};

struct Event final {
  std::uint64_t sequence{};
  Phase phase{Phase::KernelCheckProjection};
  std::uint64_t context_id{};
  std::uint64_t graph_id_hi{};
  std::uint64_t graph_id_lo{};
  std::uint64_t node_count{};
  std::uint64_t tile_count{};
  std::uint64_t binding_count{};
  std::uint32_t mode{};
  std::uint32_t failed_node{std::numeric_limits<std::uint32_t>::max()};
  bool check_ok{};
  bool raw_reason_present{};
  const char *inner_reason{};
  std::size_t inner_reason_length{};
  std::uint64_t inner_reason_code{};
  const char *effective_default{};
  std::size_t effective_default_length{};
  std::uint64_t effective_default_code{};
  const char *projected_reason{};
  std::size_t projected_reason_length{};
  std::uint64_t projected_reason_code{};
};

inline constexpr std::size_t EventCapacity = 32u;

struct Snapshot final {
  std::array<Event, EventCapacity> events{};
  std::uint32_t count{};
  std::uint64_t next_sequence{};
  bool overflow{};
};

namespace detail {

inline thread_local Snapshot *active = nullptr;

[[nodiscard]] inline std::size_t text_length(const char *const text) noexcept {
  if (text == nullptr) {
    return 0u;
  }
  std::size_t length = 0u;
  while (text[length] != '\0') {
    ++length;
  }
  return length;
}

[[nodiscard]] inline std::uint64_t
text_code(const char *const text, const std::size_t length) noexcept {
  constexpr std::uint64_t Offset = 1469598103934665603ull;
  constexpr std::uint64_t Prime = 1099511628211ull;
  std::uint64_t value = Offset;
  for (std::size_t index = 0u; index < length; ++index) {
    value ^= static_cast<unsigned char>(text[index]);
    value *= Prime;
  }
  return value;
}

inline void
Record(Phase phase, const std::uint64_t context_id,
       const std::uint64_t graph_id_hi, const std::uint64_t graph_id_lo,
       const std::uint64_t node_count, const std::uint64_t tile_count,
       const std::uint64_t binding_count, const std::uint32_t mode,
       const std::uint32_t failed_node, const bool check_ok,
       const char *const raw_reason, const char *const effective_default,
       const char *const projected_reason) noexcept {
  Snapshot *const snapshot = active;
  if (snapshot == nullptr) {
    return;
  }
  const std::uint64_t sequence = snapshot->next_sequence++;
  if (snapshot->count >= EventCapacity) {
    snapshot->overflow = true;
    return;
  }
  Event &event = snapshot->events[snapshot->count++];
  event.sequence = sequence;
  event.phase = phase;
  event.context_id = context_id;
  event.graph_id_hi = graph_id_hi;
  event.graph_id_lo = graph_id_lo;
  event.node_count = node_count;
  event.tile_count = tile_count;
  event.binding_count = binding_count;
  event.mode = mode;
  event.failed_node = failed_node;
  event.check_ok = check_ok;
  event.raw_reason_present = raw_reason != nullptr;
  event.inner_reason = raw_reason;
  event.inner_reason_length = text_length(raw_reason);
  event.inner_reason_code = text_code(raw_reason, event.inner_reason_length);
  event.effective_default = effective_default;
  event.effective_default_length = text_length(effective_default);
  event.effective_default_code =
      text_code(effective_default, event.effective_default_length);
  event.projected_reason = projected_reason;
  event.projected_reason_length = text_length(projected_reason);
  event.projected_reason_code =
      text_code(projected_reason, event.projected_reason_length);
}

} // namespace detail

class ScopedAccelCompileDiagnostic final {
public:
  ScopedAccelCompileDiagnostic() noexcept : previous_{detail::active} {
    detail::active = &snapshot_;
  }

  ScopedAccelCompileDiagnostic(const ScopedAccelCompileDiagnostic &) = delete;
  ScopedAccelCompileDiagnostic &
  operator=(const ScopedAccelCompileDiagnostic &) = delete;

  ~ScopedAccelCompileDiagnostic() { detail::active = previous_; }

  [[nodiscard]] const Snapshot &snapshot() const noexcept { return snapshot_; }

private:
  Snapshot snapshot_{};
  Snapshot *previous_{};
};

inline void
RecordKernelCheckProjection(const rund::AccelKernel &kernel,
                            const rund::AccelCheck &check,
                            const char *const projected_reason) noexcept {
  detail::Record(Phase::KernelCheckProjection, kernel.context_id,
                 kernel.graph_id_hi, kernel.graph_id_lo, kernel.node_count, 0u,
                 0u, 0u, std::numeric_limits<std::uint32_t>::max(), check.ok,
                 check.reason, "accel_kernel_invalid", projected_reason);
}

inline void RecordScratchExecutionAdmission(const rund::AccelContext &context,
                                            const rund::AccelKernel &kernel,
                                            const bool ok,
                                            const char *const reason) noexcept {
  detail::Record(Phase::ScratchExecutionAdmission, context.id,
                 kernel.graph_id_hi, kernel.graph_id_lo, kernel.node_count, 0u,
                 0u, 0u, std::numeric_limits<std::uint32_t>::max(), ok, reason,
                 "accel_kernel_scratch_invalid", nullptr);
}

inline void
RecordScratchPlanProjection(const rund::AccelKernel &kernel, const bool ok,
                            const char *const raw_reason,
                            const char *const effective_default,
                            const char *const projected_reason) noexcept {
  detail::Record(Phase::ScratchPlanProjection, kernel.context_id,
                 kernel.graph_id_hi, kernel.graph_id_lo, kernel.node_count, 0u,
                 0u, 0u, std::numeric_limits<std::uint32_t>::max(), ok,
                 raw_reason, effective_default, projected_reason);
}

inline void RecordPrepareRun(const Phase phase,
                             const rund::AccelContext &context,
                             const rund::AccelKernel &kernel,
                             const rund::AccelRun &run,
                             const std::uint32_t mode, const bool ok,
                             const char *const raw_reason,
                             const std::uint32_t failed_node) noexcept {
  detail::Record(phase, context.id, kernel.graph_id_hi, kernel.graph_id_lo,
                 kernel.node_count, run.tile_count, run.binding_count, mode,
                 failed_node, ok, raw_reason, "accel_kernel_run_invalid",
                 nullptr);
}

inline void RecordPrepareRunProjection(
    const rund::AccelContext &context, const rund::AccelKernel &kernel,
    const std::uint64_t tile_count, const std::uint64_t binding_count,
    const std::uint32_t mode, const bool ok, const char *const raw_reason,
    const char *const projected_reason) noexcept {
  detail::Record(Phase::PrepareRunProjection, context.id, kernel.graph_id_hi,
                 kernel.graph_id_lo, kernel.node_count, tile_count,
                 binding_count, mode, std::numeric_limits<std::uint32_t>::max(),
                 ok, raw_reason, "accel_kernel_run_invalid", projected_reason);
}

} // namespace rund::node::accel::diagnostic
