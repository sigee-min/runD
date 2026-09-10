#include "support.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../../src/accel/backend/token.hpp"
#include "../../../../../src/accel/vulkan/adapter/access.hpp"
#include "../../../../../src/accel/vulkan/adapter/state.hpp"
#include "../../../../../src/accel/vulkan/cached/index.hpp"
#include "../../../../../src/accel/vulkan/cached/pipeline.hpp"
#include "../../../../../src/accel/vulkan/collective/pipeline.hpp"

#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/binding/model.hpp>
#include <node/accel/pick.hpp>
#include <rund/counter.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#endif

namespace rund::node::vulkan_cache_contract {

[[nodiscard]] int VulkanCollectiveSourceIdentity() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  rund::AccelPolicy policy{};
  policy.preferred[0] = rund::AccelApi::Vulkan;
  policy.preferred_count = 1u;
  policy.allow_fake = false;
  const rund::AccelDevice pick = rund::node::accel::PickAccel(policy);
  if (!pick.check.ok) {
    return 0;
  }
  using namespace rund::node::accel::detail;
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  VulkanAdapter *const adapter =
      raw == nullptr ? nullptr : CheckedVulkanAdapter(*raw);
  if (adapter == nullptr) {
    return 15;
  }
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x8fcb4ad7211038c1ull,
      .op_hash_lo = 0x72970c9c2b1676c9ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact first_artifact{
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text =
          "#version 450\nlayout(local_size_x=1) in;\nvoid main() {}\n",
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact second_artifact{
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text = "#version 450\nlayout(local_size_x=1) in;\n"
                     "void main() { uint x = gl_GlobalInvocationID.x; }\n",
      .ok = true,
      .reason = "ok",
  };
  std::lock_guard lock{adapter->mutex};
  const std::uint64_t compile_begin = adapter->pipeline_compile_count;
  const std::uint64_t hit_begin = adapter->pipeline_cache_hit_count;
  const std::uint64_t module_begin = adapter->shader_module_create_count;
  VulkanCollectivePipeline *const first =
      AcquireVulkanCollectivePipeline(*adapter, 1u, 0u, plan, first_artifact);
  VulkanCollectivePipeline *const repeated =
      AcquireVulkanCollectivePipeline(*adapter, 1u, 0u, plan, first_artifact);
  VulkanCollectivePipeline *const distinct =
      AcquireVulkanCollectivePipeline(*adapter, 1u, 0u, plan, second_artifact);
  VulkanCollectivePipeline *const pushed = AcquireVulkanCollectivePipeline(
      *adapter, 1u, sizeof(std::uint32_t), plan, first_artifact);
  VulkanCollectivePipeline *const wider =
      AcquireVulkanCollectivePipeline(*adapter, 2u, 0u, plan, first_artifact);
  const rund::kernel::ComputePlan map_plan{
      .op_hash_hi = 0xdb530c5f26b24c11ull,
      .op_hash_lo = 0x8e4410b17a77db21ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .output_buffer_count = 1u,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact map_artifact{
      .key =
          {
              .api = rund::kernel::ComputeApi::Vulkan,
              .op_hash_hi = map_plan.op_hash_hi,
              .op_hash_lo = map_plan.op_hash_lo,
          },
      .source_text =
          "#version 450\nlayout(local_size_x=1) in;\nvoid main() {}\n",
      .ok = true,
      .reason = "ok",
  };
  VulkanCachedPipeline *const map =
      AcquireVulkanCachedPipeline(*adapter, map_plan, map_artifact);
  VulkanCachedPipeline *const map_repeated =
      AcquireVulkanCachedPipeline(*adapter, map_plan, map_artifact);
  VulkanSpecialization invalid_specialization{};
  invalid_specialization.count =
      static_cast<std::uint32_t>(invalid_specialization.values.size() + 1u);
  VulkanCollectivePipeline *const invalid = AcquireVulkanCollectivePipeline(
      *adapter, 1u, 0u, plan, first_artifact, invalid_specialization);
  if (first == nullptr) {
    return 16;
  }
  if (repeated != first) {
    return 17;
  }
  if (distinct == nullptr) {
    return 18;
  }
  if (distinct == first) {
    return 19;
  }
  if (pushed == nullptr || pushed == first ||
      pushed->push_bytes != sizeof(std::uint32_t)) {
    return 22;
  }
  if (wider == nullptr || wider == first || wider == pushed ||
      wider->descriptor_count != 2u) {
    return 25;
  }
  if (map == nullptr || map_repeated != map) {
    return 38;
  }
  if (invalid != nullptr || adapter->shader_module_current != 0u) {
    return 39;
  }
  if (::rund::detail::counter::Delta(compile_begin,
                                     adapter->pipeline_compile_count) != 5u) {
    return 20;
  }
  if (::rund::detail::counter::Delta(hit_begin,
                                     adapter->pipeline_cache_hit_count) != 2u) {
    return 21;
  }
  if (adapter->shader_module_current != 0u ||
      adapter->shader_module_peak != 1u ||
      ::rund::detail::counter::Delta(
          module_begin, adapter->shader_module_create_count) != 6u) {
    return 37;
  }
  return 0;
#else
  return 0;
#endif
}

} // namespace rund::node::vulkan_cache_contract
