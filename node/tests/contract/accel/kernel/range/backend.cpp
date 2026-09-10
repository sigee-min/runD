#include "local.hpp"

#include "src/accel/context/internal/execution.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/vulkan/kernel/manifest.hpp"
#include "src/accel/vulkan/kernel/ops/table.hpp"
#include "src/accel/vulkan/kernel/pipeline/source.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/range/local.hpp"
#include "src/accel/window/shape.hpp"

#include <kernel/program/compute/window/plan.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>

namespace node_accel_contract::range {
namespace {

using rund::kernel::ComputeDomain;
using rund::kernel::u32;
using rund::kernel::u64;
using namespace rund::node::accel::detail;

[[nodiscard]] bool ResidentControlContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangeShape prefix_shape =
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 4097u, 4097u, 8195u, 1u,
                  4097u, 4u, ComputeDomain::U32, RangeCount::U32);
  const RangePlan prefix = ContractPlanRange(
      prefix_shape, Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 4u, 32768u,
                        std::numeric_limits<u32>::max(), direct_prefix));
  const RangePlan metal_prefix = ContractPlanRange(
      prefix_shape, Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
                        std::numeric_limits<u32>::max(), direct_prefix));
  const RangePlan block = ContractPlanRange(
      AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 4097u, 4097u, 8195u,
                  1u, 4097u, 8u, ComputeDomain::I64, RangeCount::U64),
      Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct_block));
  const auto prefix_layout = RangeControlLayout::from(prefix);
  const auto block_layout = RangeControlLayout::from(block);
  const auto empty = RangeRun::make(prefix, 0u);
  const auto empty_indirect =
      empty.has_value() ? empty->indirect(0u) : std::nullopt;
  if (!prefix.ok() || !metal_prefix.ok() || !block.ok() ||
      prefix.candidate().disposition() != RangePath::PrefixDifference ||
      block.candidate().disposition() != RangePath::BlockPrefixSuffix ||
      !prefix_layout.has_value() || !block_layout.has_value() ||
      !empty_indirect.has_value() || empty_indirect->groups_x != 0u ||
      empty_indirect->groups_y != 0u || empty_indirect->groups_z != 0u ||
      empty_indirect->work_items_lo != 0u ||
      empty_indirect->work_items_hi != 0u) {
    return false;
  }
  const auto wide_block =
      ContractPlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clamp, 65u,
                                    65u, 4294967297ull, 1u, 2147483648ull, 4u,
                                    ComputeDomain::U32, RangeCount::U64),
                        Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                            std::numeric_limits<u32>::max(), direct_block));
  if (!wide_block.ok() ||
      wide_block.candidate().disposition() != RangePath::BlockPrefixSuffix ||
      wide_block.stage(0u).groups != 2u || wide_block.stage(1u).groups != 1u) {
    return false;
  }
  const auto wide_source = MetalRangeControlSource(wide_block);
  if (wide_source.find("auxiliary = 1ul + (elements - 1ul) / 4294967297ul;") ==
          std::string::npos ||
      wide_source.find("groups = (count - 1ul) / 4294967297ul + 1ul;") ==
          std::string::npos) {
    return false;
  }
  const std::string metal_source = MetalRangeControlSource(metal_prefix);
  std::uint64_t metal_bytes = 0u;
  if (metal_source.empty() ||
      !MetalRangeControlSourceUpperBytes(metal_prefix, metal_bytes) ||
      metal_bytes != metal_source.size() ||
      metal_source.find("kernel void rund_range_control") ==
          std::string::npos ||
      metal_source.find("const bool valid = count <= 4097ul;") ==
          std::string::npos ||
      metal_source.find("status[0] = uint2(valid ? 0u : 1u, 0u);") ==
          std::string::npos ||
      metal_source.find(
          "indirect[at + 1u] = valid && groups != 0ul ? 1u : 0u;") ==
          std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const std::string prefix_source = VulkanRangeControlSource(prefix);
  const std::string block_source = VulkanRangeControlSource(block);
  std::uint64_t prefix_bytes = 0u;
  std::uint64_t block_bytes = 0u;
  std::string modified = prefix_source;
  modified.push_back('\n');
  return !prefix_source.empty() && !block_source.empty() &&
         VulkanRangeControlSourceBytes(prefix, prefix_bytes) &&
         VulkanRangeControlSourceBytes(block, block_bytes) &&
         prefix_bytes == prefix_source.size() &&
         block_bytes == block_source.size() &&
         prefix_layout->params_bytes() ==
             prefix.stage_count() * sizeof(RangeParams) &&
         prefix_layout->indirect_bytes() ==
             prefix.stage_count() * sizeof(RangeIndirect) &&
         prefix_source.find("uint param_stride_words;") != std::string::npos &&
         prefix_source.find(
             "rund_range_store_params(tid * push.param_stride_words, row);") !=
             std::string::npos &&
         prefix_source.find("const bool valid = count <= 4097ul;") !=
             std::string::npos &&
         prefix_source.find("status[0] = uvec2(valid ? 0u : 1u, 0u);") !=
             std::string::npos &&
         prefix_source.find("indirect[at + 0u] = valid ? uint(groups) : 0u;") !=
             std::string::npos &&
         block_source.find(
             "uint64_t(count_words[push.count_word + 1u]) << 32u") !=
             std::string::npos &&
         block_source.find("elements = count - uint64_t(1) + 8195ul;") !=
             std::string::npos &&
         VulkanRangeControlSourceMatches(prefix, prefix_source,
                                         SourceHash(prefix_source)) &&
         VulkanRangeControlSourceMatches(block, block_source,
                                         SourceHash(block_source)) &&
         !VulkanRangeControlSourceMatches(prefix, modified,
                                          SourceHash(modified)) &&
         !VulkanRangeControlSourceMatches(block, prefix_source,
                                          SourceHash(prefix_source));
#else
  return true;
#endif
}

[[nodiscard]] bool VulkanResidentImmutableContract() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  constexpr u64 capacity = 4097u;
  const rund::kernel::WindowDesc desc{
      .op = rund::kernel::WindowOp::Sum,
      .element = rund::kernel::WindowElement::U32,
      .boundary = rund::kernel::WindowBoundary::Clamp,
      .domain = ComputeDomain::U32,
      .count_source = rund::kernel::ComputeCountSource::BufferU32,
      .input_count = capacity,
      .output_count = capacity,
      .window_size = 8195u,
      .stride = 1u,
      .pad_left = 4097u,
  };
  const rund::kernel::WindowPlan semantic = rund::kernel::PlanWindow(desc);
  const std::optional<RangeShape> shape = WindowRangeShape(semantic);
  if (!semantic.ok || !shape.has_value()) {
    return false;
  }
  constexpr std::uint8_t support =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const RangePlan range = ContractPlanRange(
      *shape, Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 4u, 32768u,
                  std::numeric_limits<u32>::max(), support));
  if (!range.ok() ||
      range.candidate().disposition() != RangePath::PrefixDifference ||
      range.stage_count() < 2u) {
    return false;
  }

  KernelExecutionStep step{};
  step.operation.set<operation::Window>(desc, semantic, range);
  step.control = rund::kernel::GraphControl{
      .count_source = rund::kernel::GraphControlSource::U32,
      .count_binding = 0u,
      .capacity = capacity,
  };
  const rund::kernel::ComputePlan compute{
      .api = rund::kernel::ComputeApi::Vulkan,
      .scalar = rund::kernel::ComputeScalar::Lane32,
      .domain = ComputeDomain::U32,
      .dispatch_count = 1u,
      .ok = true,
      .reason = "ok",
  };
  const PreparedBackendManifest manifest = BuildVulkanBackendManifest(
      step, compute, nullptr, std::numeric_limits<u32>::max());
  const u64 stages = range.stage_count();
  const u64 data_bindings = stages * RangeDescriptorCount(range);
  if (!manifest.ok || manifest.source_build_count != 2u ||
      manifest.source_library_dependency_count != 2u ||
      manifest.pipeline_stage_count != stages + 1u ||
      manifest.descriptor_set_count != stages + 1u ||
      manifest.descriptor_binding_count != data_bindings + 4u ||
      manifest.descriptor_lease_count != stages + 1u ||
      manifest.descriptor_dependency_count != stages + 1u ||
      manifest.capture_direct_dispatch_count != 1u ||
      manifest.capture_indirect_dispatch_count != stages ||
      manifest.status_source_count != 1u || manifest.status_entry_count != 1u ||
      manifest.telemetry_source_count != 1u ||
      manifest.cache_dependency_entry_count != 2u ||
      manifest.native_pipeline_dependency_count != 2u ||
      manifest.cold_cache_native_object_count != stages + 9u ||
      manifest.source_dependencies[0u].pipeline_stage_count != 1u ||
      manifest.source_dependencies[1u].pipeline_stage_count != 1u ||
      manifest.source_dependencies[0u].source_recipe != 0x76756c6b2e726e67ull ||
      manifest.source_dependencies[1u].source_recipe != 0x76756c6b2e726374ull) {
    return false;
  }

  const std::string_view telemetry_source = VulkanTelemetrySourceText();
  const std::string profile_source = VulkanProfileSource();
  if (telemetry_source.find("if (p.kind == 5u)") == std::string_view::npos ||
      telemetry_source.find("index += 8u") == std::string_view::npos ||
      telemetry_source.find(
          "pair64(primary[index + 3u], primary[index + 4u])") ==
          std::string_view::npos ||
      profile_source.find("if (p.kind == 5u)") == std::string::npos ||
      profile_source.find("index += 8u") == std::string::npos) {
    return false;
  }

  VulkanRangeResources resources{};
  VulkanBuffer count_buffer{};
  count_buffer.buffer = reinterpret_cast<VkBuffer>(&resources);
  resources.range = range;
  resources.stage_count = static_cast<std::uint32_t>(stages);
  resources.controlled = true;
  resources.control = step.control;
  resources.control.count_byte_offset = 4u;
  resources.control_count.device_buffer = &count_buffer;
  resources.control_count.ref.offset_bytes = 8u;
  resources.control_indirect.buffer = reinterpret_cast<VkBuffer>(&count_buffer);
  const std::shared_ptr<void> resource_owner{&resources, [](void *) {}};
  VulkanPipelineTelemetrySource telemetry{};
  const rund::AccelCheck telemetry_check =
      DescribeVulkanRangePipelineTelemetry(resource_owner, telemetry);
  if (!telemetry_check.ok ||
      telemetry.kind != VulkanPipelineTelemetryKind::ControlledRange ||
      telemetry.primary != &resources.control_indirect ||
      telemetry.count != &count_buffer || telemetry.control.iteration != 0u ||
      telemetry.count_offset != 12u || telemetry.capacity != capacity ||
      telemetry.primary_word_count != stages * 8u ||
      telemetry.indirect_dispatch_count != stages) {
    return false;
  }

  VulkanCollectivePipeline control{};
  VulkanCollectivePipeline data{};
  control.descriptor_count = 4u;
  data.descriptor_count = RangeDescriptorCount(range);
  VulkanKernelImmutablePipelines immutable{};
  immutable.kind = rund::kernel::NodeKind::Window;
  immutable.capture_direct_dispatch_count = 1u;
  immutable.capture_indirect_dispatch_count = stages;
  if (!immutable.append_control(&control, 4u, 1u)) {
    return false;
  }
  for (std::size_t index = 0u; index < stages; ++index) {
    if (!immutable.append(&data, data.descriptor_count, 1u)) {
      return false;
    }
  }
  PreparedBackendManifest wrong = manifest;
  --wrong.capture_indirect_dispatch_count;
  return immutable.ready(rund::kernel::NodeKind::Window, manifest) &&
         immutable.borrow_control(rund::kernel::NodeKind::Window, 4u, 1u) ==
             &control &&
         immutable.borrow(rund::kernel::NodeKind::Window,
                          static_cast<std::uint32_t>(stages), stages - 1u,
                          data.descriptor_count, 1u) == &data &&
         !immutable.ready(rund::kernel::NodeKind::Window, wrong);
#else
  return true;
#endif
}

[[nodiscard]] bool ManifestCompletionContract() {
  const auto make = [] {
    PreparedBackendManifest manifest{
        .source_build_count = 1u,
        .source_library_dependency_count = 1u,
        .pipeline_stage_count = 1u,
        .descriptor_set_count = 1u,
        .descriptor_dependency_count = 1u,
        .capture_direct_dispatch_count = 1u,
    };
    return AddPreparedBackendCacheDependency(manifest,
                                             PreparedBackendCacheDependency{
                                                 .source_recipe = 1u,
                                                 .source_upper_bytes = 1u,
                                                 .pipeline_stage_count = 1u,
                                             })
               ? manifest
               : PreparedBackendManifest{};
  };
  PreparedBackendManifest valid = make();
  PreparedBackendManifest no_build = make();
  no_build.source_build_count = 0u;
  PreparedBackendManifest build_mismatch = make();
  build_mismatch.source_build_count = 2u;
  PreparedBackendManifest aliased_stages = make();
  aliased_stages.pipeline_stage_count = 3u;
  aliased_stages.descriptor_set_count = 3u;
  aliased_stages.descriptor_dependency_count = 3u;
  aliased_stages.capture_direct_dispatch_count = 3u;
  PreparedBackendManifest dependency_mismatch = make();
  dependency_mismatch.source_dependencies[0u].pipeline_stage_count = 2u;
  return CompleteVulkanBackendManifest(valid) && valid.ok &&
         valid.native_pipeline_dependency_count == 1u &&
         valid.cold_cache_native_object_count == 5u &&
         !CompleteVulkanBackendManifest(no_build) && !no_build.ok &&
         !CompleteVulkanBackendManifest(build_mismatch) && !build_mismatch.ok &&
         CompleteVulkanBackendManifest(aliased_stages) && aliased_stages.ok &&
         aliased_stages.native_pipeline_dependency_count == 1u &&
         aliased_stages.cold_cache_native_object_count == 7u &&
         !CompleteVulkanBackendManifest(dependency_mismatch) &&
         !dependency_mismatch.ok &&
         dependency_mismatch.native_pipeline_dependency_count == 0u &&
         dependency_mismatch.cold_cache_native_object_count == 0u;
}

} // namespace

bool BackendContract() {
  return ResidentControlContract() && VulkanResidentImmutableContract() &&
         ManifestCompletionContract();
}

} // namespace node_accel_contract::range
