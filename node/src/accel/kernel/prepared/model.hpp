#pragma once

#include "callback.hpp"
#include "../memory.hpp"
#include "../preparation.hpp"
#include "../status.hpp"

#include "../../backend/ops/table.hpp"
#include "../../context/admission/local.hpp"
#include "../../context/internal.hpp"
#include "../roundtrip.hpp"
#include "../run/bindings.hpp"
#include "../run/dispatch.hpp"
#include "../schedule.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>

namespace rund::node::accel::detail {

struct ServiceFreeDirectProof;

namespace prepared {

struct RunState;
struct PipelineState;

struct RunSubmission final {
  std::mutex mutex{};
  rund::AccelContext context{};
  RunState *prepared{};
  std::shared_ptr<void> owner{};
  std::shared_ptr<void> lifetime{};
  PreparedKernelCompletion completion{};
  void *user{};
  bool active{};
};

struct RunState final {
  KernelExecution execution{};
  std::uint64_t tile_count{};
  RunBindBuild binds{};
  ResetBindBuild resets{};
  RunDispatchBuild dispatch{};
  ScheduledStepOrder order{};
  ProducerConsumerRoundtrip roundtrip{};
  BoundRun bound{};
  KernelPreparationMode mode{KernelPreparationMode::Standalone};
  RunSubmission submission{};
  mutable PreparedMemoryMeter memory{};
  // Backend resources may view every common owner above and must die first.
  std::shared_ptr<void> backend{};
};

inline constexpr std::size_t BatchCapacity = 64u;

struct PipelineSubmission final {
  std::mutex mutex{};
  std::shared_ptr<void> owner{};
  std::shared_ptr<void> lifetime{};
  PreparedPipelineCompletion completion{};
  void *user{};
  const std::uint32_t *selected_steps{};
  std::size_t selected_step_count{};
  // Unknown native terminal deliberately retains owner+lifetime as a
  // self-quarantine. The commands may still write them, so destruction or a
  // later submission would be use-after-submit rather than cleanup.
  bool quarantined{};
  // A bounded residency window claims a prepared Pipeline once even when the
  // same two-bank owner occurs twice in its fixed four-entry request.
  void *window{};
  // Whole-stream peer exclusion and strong-owner sentinel. Intermediate
  // bounded Finals clear `window` but never this claim.
  void *stream{};
  // Whole-run schedule claim. It is distinct from a bounded raw window so an
  // adapter cannot reinterpret a schedule terminal through W4 storage.
  void *schedule{};
  // Fixed-state sliding invocation claim. Individual selected submissions
  // may come and go, but peer Pipeline users remain excluded until the one
  // invocation Final or a sticky Unknown quarantine.
  void *sliding{};

  [[nodiscard]] bool active() const noexcept {
    return owner != nullptr || window != nullptr || stream != nullptr ||
           schedule != nullptr || sliding != nullptr;
  }
  [[nodiscard]] PipelineState *pipeline() const noexcept {
    return static_cast<PipelineState *>(owner.get());
  }
};

struct EvidenceCounts final {
  std::uint64_t original_operations{};
  std::uint64_t fused_operations{};
  std::uint64_t original_dispatches{};
  std::uint64_t final_dispatches{};
  std::uint64_t fusion_rejections{};
  std::uint64_t internal_roundtrip_bytes{};
  std::uint64_t external_roundtrip_bytes{};
};

struct PipelineState final {
  rund::AccelContext context{};
  const BackendOps *ops{};
  std::unique_ptr<std::shared_ptr<RunState>[]> states{};
  // Keeps immutable Program templates alive across both the frozen backend
  // command stream and every route resource that views them.
  std::shared_ptr<void> templates{};
  PreparedPipelineStatusLayout status{};
  mutable PreparedPipelineMemoryMeter memory{};
  // Backend resources may view the memory meter and must die first.
  std::shared_ptr<void> backend{};
  // Common semantic proof for a whole-pipeline resident Map recurrence. The
  // proof retains its exact RunState authority and never points back to this
  // PipelineState, so Prepared/request lifetime cannot form a self-cycle.
  std::shared_ptr<const ServiceFreeDirectProof> service_free_direct{};
  PipelineSubmission submission{};
  EvidenceCounts counts{};
  // Physical strong-owner rows retained by `states`. Ordinary Pipelines keep
  // one row per declared template. A proved service-free recurrence compacts
  // repeated references to its one semantic RunState after backend
  // preparation; `size` remains the logical authored-step count used by
  // aggregate evidence.
  std::size_t state_count{};
  std::size_t size{};
};

[[nodiscard]] inline bool MatchesContext(const rund::AccelContext &context,
                                         const RunState &state) noexcept {
  return state.execution.admission.check.ok &&
         ContextMatchesAdmission(context, state.execution.context_admission);
}

[[nodiscard]] inline bool
ValidPipeline(const rund::AccelContext &context,
              const PipelineState &pipeline) noexcept {
  return pipeline.ops != nullptr && pipeline.backend != nullptr &&
         pipeline.size != 0u && pipeline.state_count != 0u &&
         pipeline.states != nullptr && pipeline.states[0] != nullptr &&
         MatchesContext(context, *pipeline.states[0]);
}

} // namespace prepared
} // namespace rund::node::accel::detail
