#pragma once

#include "../prepared.hpp"

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

namespace rund::node::accel::detail::prepared {

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

  [[nodiscard]] bool active() const noexcept { return owner != nullptr; }
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
  std::shared_ptr<void> backend{};
  PreparedPipelineMemory memory{};
  PipelineSubmission submission{};
  EvidenceCounts counts{};
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
         pipeline.size != 0u && pipeline.states != nullptr &&
         pipeline.states[0] != nullptr &&
         MatchesContext(context, *pipeline.states[0]);
}

} // namespace rund::node::accel::detail::prepared
