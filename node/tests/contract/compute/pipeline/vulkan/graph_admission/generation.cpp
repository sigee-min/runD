#include "local.hpp"

#if defined(RUND_NODE_TEST_BACKEND_CPU) ||                                     \
    defined(RUND_NODE_TEST_BACKEND_METAL) ||                                   \
    !defined(RUND_NODE_HAVE_VULKAN_SDK)

namespace node_compute_pipeline_vulkan_graph_admission {

bool CheckGenerationSeedDomains(
    const std::shared_ptr<rund::compute::detail::PipelineState> &) {
  return false;
}

} // namespace node_compute_pipeline_vulkan_graph_admission

#else

#include "src/accel/kernel/prepared/model.hpp"
#include "src/accel/vulkan/kernel.hpp"
#include "src/accel/vulkan/kernel/pipeline/prepare/record.hpp"
#include "src/accel/vulkan/kernel/pipeline/residency/mode.hpp"
#include "src/accel/vulkan/kernel/pipeline/state.hpp"
#include "src/compute/virtual/state.hpp"

#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>

namespace node_compute_pipeline_vulkan_graph_admission {
namespace {

using rund::node::accel::detail::VulkanPipeline;

[[nodiscard]] VulkanPipeline *
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
             : static_cast<VulkanPipeline *>(prepared->backend.get());
}

[[nodiscard]] bool IsInvalid(const rund::AccelCheck result) noexcept {
  return !result.ok && result.reason != nullptr &&
         std::string_view{result.reason} == "accel_kernel_pipeline_invalid";
}

[[nodiscard]] bool
CheckSeed(VulkanPipeline &fixture,
          const rund::node::accel::detail::VulkanResidencyMode mode,
          const std::uint32_t seed, const bool expected_ok,
          const std::uint64_t expected_generation,
          const bool expect_unchanged) noexcept {
  using namespace rund::node::accel::detail;
  fixture.expected_control_generation = 0x1122334455667788ull;
  fixture.residency.reset();
  fixture.residency = std::make_shared<VulkanResidencySelection>();
  fixture.residency->mode = mode;
  const std::uint64_t before = fixture.expected_control_generation;
  const rund::AccelCheck result = SeedPreparedVulkanPipelineGeneration(
      std::shared_ptr<void>{&fixture, [](void *) {}}, seed);
  if (result.ok != expected_ok) {
    return false;
  }
  if (!expected_ok && !IsInvalid(result)) {
    return false;
  }
  if (expect_unchanged && fixture.expected_control_generation != before) {
    return false;
  }
  if (!expect_unchanged &&
      fixture.expected_control_generation != expected_generation) {
    return false;
  }
  return true;
}

} // namespace

bool CheckGenerationSeedDomains(
    const std::shared_ptr<rund::compute::detail::PipelineState> &state) {
  using namespace rund::node::accel::detail;
  VulkanPipeline *const source = NativePipeline(state);
  if (source == nullptr || source->adapter == nullptr) {
    return false;
  }

  VulkanPipeline fixture;
  fixture.adapter = source->adapter;
  fixture.dispatch_count = 0u;

  constexpr std::uint32_t max_seed = std::numeric_limits<std::uint32_t>::max();
  if (!CheckSeed(fixture, VulkanResidencyMode::Direct, max_seed, true,
                 static_cast<std::uint64_t>(max_seed) + 1u, false) ||
      !CheckSeed(fixture, VulkanResidencyMode::GraphStageGeneratedIndirect,
                 max_seed, false, 0u, true) ||
      !CheckSeed(fixture, VulkanResidencyMode::GraphStageSequence, max_seed,
                 false, 0u, true) ||
      !CheckSeed(fixture, VulkanResidencyMode::GraphStageGeneratedIndirect, 7u,
                 true, 8u, false) ||
      !CheckSeed(fixture, VulkanResidencyMode::GraphStageSequence, 7u, true, 8u,
                 false)) {
    return false;
  }

  fixture.residency.reset();
  fixture.expected_control_generation = 0x1122334455667788ull;
  const std::uint64_t before = fixture.expected_control_generation;
  const rund::AccelCheck ordinary = SeedPreparedVulkanPipelineGeneration(
      std::shared_ptr<void>{&fixture, [](void *) {}}, max_seed);
  return ordinary.ok && fixture.expected_control_generation != before &&
         fixture.expected_control_generation ==
             static_cast<std::uint64_t>(max_seed) + 1u;
}

} // namespace node_compute_pipeline_vulkan_graph_admission

#endif
