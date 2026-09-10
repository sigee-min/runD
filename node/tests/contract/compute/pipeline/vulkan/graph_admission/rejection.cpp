#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_METAL) ||                                   \
    !defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace node_compute_pipeline_vulkan_graph_admission {

bool PreparedPipelineReleaseUnlocks(
    const std::shared_ptr<rund::compute::detail::PipelineState> &) {
  return false;
}

[[nodiscard]] bool RejectRecordedGraphReuse(
    const std::shared_ptr<rund::compute::detail::PipelineState> &) {
  return false;
}

[[nodiscard]] bool RejectRecordedGraphReprepare(
    const std::shared_ptr<rund::compute::detail::PipelineState> &) {
  return false;
}

} // namespace node_compute_pipeline_vulkan_graph_admission

#else

#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/kernel/pipeline/prepare/record.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/generated_indirect/internal.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/local.hpp"
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <thread>

namespace node_compute_pipeline_vulkan_graph_admission {
namespace {

[[nodiscard]] rund::node::accel::detail::VulkanPipeline *
NativePipeline(const std::shared_ptr<rund::compute::detail::PipelineState>
                   &state) noexcept {
  if (state == nullptr || state->prepared.owner == nullptr) {
    return nullptr;
  }
  const auto *const prepared =
      static_cast<const rund::node::accel::detail::prepared::PipelineState *>(
          state->prepared.owner.get());
  return prepared == nullptr
             ? nullptr
             : static_cast<rund::node::accel::detail::VulkanPipeline *>(
                   prepared->backend.get());
}

} // namespace

bool PreparedPipelineReleaseUnlocks(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state) {
  using namespace rund::node::accel::detail;
  auto *const native = NativePipeline(state);
  if (native == nullptr || native->adapter == nullptr) {
    return false;
  }
  auto &adapter = *native->adapter;
  for (const bool failed : {false, true}) {
    bool released = false;
    bool unlocked = false;
    auto pipeline = std::make_shared<VulkanPipeline>();
    pipeline->adapter = &adapter;
    pipeline->record = std::make_unique<VulkanPipelineRecordRecipe>();
    pipeline->record->entries.push_back(VulkanPipelineRecordEntry{
        .prepared = std::shared_ptr<void>{
            new int{}, [&adapter, &released, &unlocked](void *value) {
              released = true;
          // Check from another thread: try_lock on a mutex already owned by
          // this thread would itself be undefined in the regressed path.
          std::thread observer{[&] {
            unlocked = adapter.mutex.try_lock();
            if (unlocked) {
              adapter.mutex.unlock();
            }
          }};
          observer.join();
              delete static_cast<int *>(value);
            }}});
    if (failed) {
      // Keep the mutex object alive even if a regressed failure helper resets
      // its argument; the assertions still reject that premature reset.
      const auto guard_owner = pipeline;
      std::scoped_lock lock{guard_owner->submission.mutex, adapter.mutex};
      const auto result =
          FailVulkanPipeline(pipeline, "compute_pipeline_capacity");
      if (result.ok || pipeline == nullptr || released ||
          pipeline->adapter != nullptr) {
        return false;
      }
    }
    pipeline.reset();
    if (!released || !unlocked) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool RejectRecordedGraphReuse(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state) {
  using namespace rund::node::accel::detail;
  auto *const native = NativePipeline(state);
  if (native == nullptr || native->residency == nullptr ||
      native->residency->mode !=
          VulkanResidencyMode::GraphStageGeneratedIndirect ||
      native->residency->graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::Recorded ||
      native->residency->graph_generated.local_count == 0u) {
    return false;
  }
  auto &selection = *native->residency;
  std::array<std::uint32_t, PreparedPipelineStepCapacity> locals{};
  for (std::size_t index = 0u; index < selection.graph_generated.local_count;
       ++index) {
    locals[index] = static_cast<std::uint32_t>(index);
  }
  const VkCommandBuffer before =
      selection.graph_generated_frames[0u].command.buffer;
  const std::size_t count = selection.graph_generated_command_count;
  const bool accepted = vulkan_generated_indirect_detail::record(
      *native, selection,
      std::span<const std::uint32_t>{locals.data(),
                                     selection.graph_generated.local_count});
  return !accepted &&
         selection.graph_generated_phase ==
             VulkanResidencyGraphGeneratedPhase::Quarantined &&
         selection.graph_generated_command_count == count &&
         selection.graph_generated_frames[0u].command.buffer == before;
}

[[nodiscard]] bool RejectRecordedGraphReprepare(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state) {
  using namespace rund::node::accel::detail;
  auto *const native = NativePipeline(state);
  if (native == nullptr || native->residency == nullptr ||
      native->residency->mode !=
          VulkanResidencyMode::GraphStageGeneratedIndirect ||
      native->residency->graph_generated_phase !=
          VulkanResidencyGraphGeneratedPhase::Recorded ||
      native->residency->graph_generated_generation == 0u) {
    return false;
  }
  auto &selection = *native->residency;
  const std::uint64_t generation = selection.graph_generated_generation;
  const bool accepted =
      vulkan_generated_indirect_detail::prepare(*native, selection).ok;
  return !accepted &&
         selection.graph_generated_phase ==
             VulkanResidencyGraphGeneratedPhase::Quarantined &&
         selection.graph_generated_generation == generation &&
         selection.graph_generated_command_count == 0u;
}

} // namespace node_compute_pipeline_vulkan_graph_admission

#endif
