#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "src/accel/context/internal/support.hpp"
#include "src/accel/kernel/preparation.hpp"
#include "src/accel/metal/pipeline/cache.hpp"
#include "src/accel/metal/pipeline/template.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/stencil/shape.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/range/local.hpp"
#include "stencil/local.hpp"
#include "stencil/match/run.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/pick.hpp>

#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool StencilMatch(const bool ok, const char *const name) {
  if (ok) {
    return true;
  }
  std::cerr << "stencil backend match failed: " << name << '\n';
  return false;
}

[[nodiscard]] rund::node::accel::detail::RangePlan
MetalStencilRangePlan(const rund::AccelDevice &pick,
                      const rund::kernel::StencilPlan &plan,
                      const rund::kernel::ComputeDomain domain) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape = StencilRangeShape(plan, domain);
  return shape.has_value() ? PlanRange(*shape, MetalRangeCaps(pick))
                           : RangePlan::rejected("accel_kernel_graph_invalid");
}

[[nodiscard, maybe_unused]] rund::node::accel::detail::RangePlan
VulkanStencilRangePlan(const rund::AccelDevice &pick,
                       const rund::kernel::StencilPlan &plan,
                       const rund::kernel::ComputeDomain domain) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeShape> shape = StencilRangeShape(plan, domain);
  return shape.has_value() ? PlanRange(*shape, VulkanRangeCaps(pick))
                           : RangePlan::rejected("accel_kernel_graph_invalid");
}

[[nodiscard]] bool SignedStencilSourcesCarryDomainOrder() {
  using namespace rund::node::accel::detail;
  constexpr RangeGpuShape requested = stencil::RangeSharedShape(64u, 64u);
  constexpr RangePlan metal_range = stencil::PlanStencilSourceVariant(
      RangeSource::Metal, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, requested,
      stencil::SourcePlanPath::SharedHalo);
  static_assert(metal_range.ok());
  const std::string metal = rund::node::accel::detail::MetalRangeSource(
      stencil::RequireRangeExec(metal_range));
  if (metal.find("rund_range_min_i32") == std::string::npos ||
      metal.find("device const int* input") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  constexpr RangePlan vulkan_range = stencil::PlanStencilSourceVariant(
      RangeSource::Vulkan, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, requested,
      stencil::SourcePlanPath::SharedHalo);
  static_assert(vulkan_range.ok());
  const std::string vulkan = rund::node::accel::detail::VulkanRangeSource(
      stencil::RequireRangeExec(vulkan_range));
  if (vulkan.find("int value = range_tile[first]") == std::string::npos) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] constexpr bool RangeDispatchIsExact() noexcept {
  using namespace rund::node::accel::detail;
  constexpr rund::kernel::u64 u32_groups =
      std::numeric_limits<rund::kernel::u32>::max();
  for (const rund::kernel::u32 width : kRangeWidths) {
    const RangeGpuShape shape = stencil::RangeSharedShape(width, width);
    const rund::kernel::u64 u32_group_elements = u32_groups * width;
    const rund::kernel::u64 vulkan_u32_groups = RangePhysicalGroupCount(
        std::numeric_limits<rund::kernel::u32>::max(), shape);
    if (shape.shared_element_capacity() != 3u * width ||
        RangePhysicalGroupCount(0u, shape) != 0u ||
        RangePhysicalGroupCount(1u, shape) != 1u ||
        RangePhysicalGroupCount(width, shape) != 1u ||
        RangePhysicalGroupCount(width + 1u, shape) != 2u ||
        !RangePhysicalGroupsFit(7u * width, 7u, shape) ||
        RangePhysicalGroupsFit(7u * width + 1u, 7u, shape) ||
        !RangePhysicalGroupsFit(u32_group_elements, u32_groups, shape) ||
        RangePhysicalGroupsFit(u32_group_elements + 1u, u32_groups, shape) ||
        RangePhysicalGroupsFit(u32_group_elements + 1u, u32_groups + 1u,
                               shape) ||
        !RangeVulkanDispatchFits(std::numeric_limits<rund::kernel::u32>::max(),
                                 vulkan_u32_groups, shape) ||
        RangeVulkanDispatchFits(
            static_cast<rund::kernel::u64>(
                std::numeric_limits<rund::kernel::u32>::max()) +
                1u,
            vulkan_u32_groups + 1u, shape) ||
        RangePhysicalGroupsFit(1u, 0u, shape)) {
      return false;
    }
  }
  return true;
}

static_assert(RangeDispatchIsExact());

[[nodiscard]] constexpr bool RangePlanProjectionIsExact() noexcept {
  using namespace rund::node::accel::detail;
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  constexpr std::uint8_t direct_shared =
      direct | RangeSupportBit(RangeSupport::SharedHalo);
  constexpr std::uint8_t direct_prefix =
      direct | RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  constexpr auto shared_capabilities = RangeCaps::gpu(
      RangeSource::Vulkan, kRangeWidth128Bit, 128u, 4u,
      (128u + 2u * 7u) * 8u * 4u, std::numeric_limits<rund::kernel::u32>::max(),
      std::numeric_limits<rund::kernel::u32>::max(), direct_shared);
  constexpr auto direct_capabilities =
      RangeCaps::gpu(RangeSource::Metal, kRangeWidth128Bit, 128u, 0u, 0u,
                     std::numeric_limits<rund::kernel::u32>::max(),
                     std::numeric_limits<rund::kernel::u64>::max(), direct);
  constexpr auto prefix_capabilities = RangeCaps::gpu(
      RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
      std::numeric_limits<rund::kernel::u32>::max(),
      std::numeric_limits<rund::kernel::u64>::max(), direct_prefix);
  constexpr auto block_capabilities = RangeCaps::gpu(
      RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
      std::numeric_limits<rund::kernel::u32>::max(),
      std::numeric_limits<rund::kernel::u32>::max(), direct_block);
  constexpr auto sum_u64 =
      RangeTraits::sum_modulo(rund::kernel::ComputeDomain::U64);
  constexpr auto sum_u32 =
      RangeTraits::sum_modulo(rund::kernel::ComputeDomain::U32);
  constexpr auto minimum =
      RangeTraits::minimum(rund::kernel::ComputeDomain::I32);
  constexpr auto shared_shape =
      RangeShape::window(*sum_u64, RangeBoundary::Clamp, 129u, 7u, 8u);
  constexpr auto direct_shape =
      RangeShape::window(*sum_u32, RangeBoundary::Clamp, 65u, 65u, 4u);
  constexpr auto prefix_shape =
      RangeShape::window(*sum_u32, RangeBoundary::Clamp, 515u, 515u, 4u);
  constexpr auto block_shape =
      RangeShape::window(*minimum, RangeBoundary::Clamp, 515u, 515u, 4u);
  constexpr RangePlan shared_plan =
      PlanRange(*shared_shape, *shared_capabilities);
  constexpr RangePlan direct_plan =
      PlanRange(*direct_shape, *direct_capabilities);
  constexpr RangePlan prefix_plan =
      PlanRange(*prefix_shape, *prefix_capabilities);
  constexpr RangePlan block_plan = PlanRange(*block_shape, *block_capabilities);
  constexpr RangePlan cpu_plan = PlanRange(*direct_shape, RangeCaps::cpu());
  constexpr RangePlan unavailable_plan =
      PlanRange(*direct_shape, RangeCaps::unavailable());
  constexpr RangePlan metal_direct_range = stencil::PlanStencilSourceVariant(
      RangeSource::Metal, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, stencil::RangeDirectShape(64u),
      stencil::SourcePlanPath::Direct);

  return shared_plan.ok() && direct_plan.ok() && prefix_plan.ok() &&
         block_plan.ok() && cpu_plan.ok() && !unavailable_plan.ok() &&
         stencil::RequireRangeShape(shared_plan) ==
             stencil::RangeSharedShape(128u, 7u) &&
         stencil::RequireRangeShape(direct_plan) ==
             stencil::RangeDirectShape(128u) &&
         stencil::RequireRangeShape(prefix_plan) ==
             stencil::RangeDirectShape(64u) &&
         stencil::RequireRangeShape(block_plan) ==
             stencil::RangeDirectShape(64u) &&
         !RangeGpuShapeFor(cpu_plan).has_value() &&
         !RangeGpuShapeFor(unavailable_plan).has_value() &&
         StencilElementBytes(rund::kernel::StencilElement::U32) == 4u &&
         StencilElementBytes(rund::kernel::StencilElement::U64) == 8u &&
         StencilElementBytes(static_cast<rund::kernel::StencilElement>(0u)) ==
             0u &&
         MetalRangeSupports(stencil::RequireRangeExec(metal_direct_range),
                            MetalRangeLimits{
                                .maximum_workgroup_width = 64u,
                                .static_shared_bytes = 4u,
                                .shared_memory_limit = 32768u,
                            }) == MetalRangeSupport::Invalid;
}

static_assert(RangePlanProjectionIsExact());

[[nodiscard]] bool StencilShapeRejectsOnlyOverlappingStorage() {
  using namespace rund::node::accel::detail;
  constexpr rund::kernel::StencilDesc desc{
      .op = rund::kernel::StencilOp::Sum,
      .element = rund::kernel::StencilElement::U32,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = 4u,
      .radius = 1u,
  };
  constexpr rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  static_assert(plan.ok);
  rund::kernel::ResidentBufferRef input{
      .id = 41u,
      .bytes = 32u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  rund::kernel::ResidentBufferRef output{
      .id = input.id,
      .bytes = input.bytes,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  const std::shared_ptr<void> owner = std::make_shared<int>(1);
  const RangeBinds bindings{
      .input = &input,
      .input_handle = &owner,
      .output = &output,
      .output_handle = &owner,
  };
  if (StencilShapeOk(desc, plan, bindings)) {
    return false;
  }
  output.offset_bytes = 16u;
  return StencilShapeOk(desc, plan, bindings);
}

[[nodiscard]] bool MetalStencilRejectedCompileTelemetryIsExact() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  RecordMetalUncachedLibraryCompile(adapter, 17u);
  RecordMetalUncachedPipelineCompile(adapter, 23u);
  return adapter.stats.library_compile_count == 1u &&
         adapter.stats.shader_compile_ns == 17u &&
         adapter.stats.pipeline_compile_count == 1u &&
         adapter.stats.pipeline_create_ns == 23u &&
         adapter.source_libraries.empty() && adapter.named_pipelines.empty();
}

[[nodiscard]] bool MetalStencilNamedPipelinePublicationIsTransactional() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  const std::shared_ptr<void> first = std::make_shared<int>(1);
  const std::shared_ptr<void> contender = std::make_shared<int>(2);
  const MetalNamedPipelinePublishResult inserted =
      PublishMetalNamedPipeline(adapter, "stencil.contract", first, 11u);
  if (inserted.status != MetalNamedPipelinePublishStatus::Inserted ||
      inserted.pipeline != first || adapter.named_pipelines.size() != 1u ||
      adapter.stats.pipeline_compile_count != 1u ||
      adapter.stats.pipeline_create_ns != 11u) {
    return false;
  }
  const MetalNamedPipelinePublishResult existing =
      PublishMetalNamedPipeline(adapter, "stencil.contract", contender, 17u);
  if (existing.status != MetalNamedPipelinePublishStatus::Existing ||
      existing.pipeline != first || adapter.named_pipelines.size() != 1u ||
      adapter.stats.pipeline_compile_count != 1u ||
      adapter.stats.pipeline_create_ns != 11u) {
    return false;
  }
  adapter.fault_named_pipeline_publish_once.store(true,
                                                  std::memory_order_relaxed);
  const MetalNamedPipelinePublishResult failed = PublishMetalNamedPipeline(
      adapter, "stencil.capacity", std::make_shared<int>(3), 23u);
  return failed.status == MetalNamedPipelinePublishStatus::Failed &&
         failed.pipeline == nullptr && adapter.named_pipelines.size() == 1u &&
         adapter.stats.pipeline_compile_count == 1u &&
         adapter.stats.pipeline_create_ns == 11u;
}

[[nodiscard]] bool MetalRangeSourcePublicationIsTransactional() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  const std::shared_ptr<void> baseline_owner = std::make_shared<int>(0);
  const MetalSourceLibraryPublishResult baseline = PublishMetalSourceLibrary(
      adapter, "stencil.baseline", baseline_owner, 5u);
  if (baseline.status != MetalSourceLibraryPublishStatus::Inserted ||
      baseline.library != baseline_owner) {
    return false;
  }
  const std::shared_ptr<void> failed_owner = std::make_shared<int>(1);
  adapter.fault_source_library_publish_once.store(true,
                                                  std::memory_order_relaxed);
  const MetalSourceLibraryPublishResult failed =
      PublishMetalSourceLibrary(adapter, "stencil.source", failed_owner, 7u);
  if (failed.status != MetalSourceLibraryPublishStatus::Failed ||
      failed.library != nullptr || adapter.source_libraries.size() != 1u ||
      adapter.source_libraries.front().library != baseline_owner ||
      adapter.stats.library_compile_count != 2u ||
      adapter.stats.shader_compile_ns != 12u ||
      adapter.stats.library_cache_hit_count != 0u ||
      std::string_view{adapter.last_error} != "compute_pipeline_capacity") {
    return false;
  }

  const std::shared_ptr<void> inserted_owner = std::make_shared<int>(2);
  const MetalSourceLibraryPublishResult inserted =
      PublishMetalSourceLibrary(adapter, "stencil.source", inserted_owner, 11u);
  if (inserted.status != MetalSourceLibraryPublishStatus::Inserted ||
      inserted.library != inserted_owner ||
      adapter.source_libraries.size() != 2u ||
      adapter.stats.library_compile_count != 3u ||
      adapter.stats.shader_compile_ns != 23u ||
      adapter.stats.library_cache_hit_count != 0u) {
    return false;
  }

  const MetalSourceLibraryPublishResult existing = PublishMetalSourceLibrary(
      adapter, "stencil.source", std::make_shared<int>(3), 13u);
  return existing.status == MetalSourceLibraryPublishStatus::Existing &&
         existing.library == inserted_owner &&
         adapter.source_libraries.size() == 2u &&
         adapter.stats.library_compile_count == 4u &&
         adapter.stats.shader_compile_ns == 36u &&
         adapter.stats.library_cache_hit_count == 1u;
}

[[nodiscard]] bool MetalRangeSourceRetryIsExact() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  const std::shared_ptr<void> library = std::make_shared<int>(1);
  const MetalSourceLibraryPublishResult source =
      PublishMetalSourceLibrary(adapter, "stencil.retry.source", library, 11u);
  if (source.status != MetalSourceLibraryPublishStatus::Inserted ||
      source.library != library) {
    return false;
  }

  adapter.fault_named_pipeline_publish_once.store(true,
                                                  std::memory_order_relaxed);
  const MetalNamedPipelinePublishResult failed = PublishMetalNamedPipeline(
      adapter, "stencil.retry.pipeline", std::make_shared<int>(2), 13u);
  if (failed.status != MetalNamedPipelinePublishStatus::Failed ||
      failed.pipeline != nullptr || !adapter.named_pipelines.empty() ||
      adapter.source_libraries.size() != 1u ||
      adapter.stats.pipeline_compile_count != 0u) {
    return false;
  }
  // This is the CompileMetalRange caller's exact failed-publication
  // disposition: the constructed PSO is transient but still counted once.
  RecordMetalUncachedPipelineCompile(adapter, 13u);

  if (LookupMetalSourceLibrary(adapter, "stencil.retry.source") != library) {
    return false;
  }
  const std::shared_ptr<void> retry_pipeline = std::make_shared<int>(3);
  const MetalNamedPipelinePublishResult retried = PublishMetalNamedPipeline(
      adapter, "stencil.retry.pipeline", retry_pipeline, 17u);
  return retried.status == MetalNamedPipelinePublishStatus::Inserted &&
         retried.pipeline == retry_pipeline &&
         adapter.source_libraries.size() == 1u &&
         adapter.named_pipelines.size() == 1u &&
         adapter.stats.library_compile_count == 1u &&
         adapter.stats.library_cache_hit_count == 1u &&
         adapter.stats.shader_compile_ns == 11u &&
         adapter.stats.pipeline_compile_count == 2u &&
         adapter.stats.pipeline_create_ns == 30u;
}

enum class RuntimeSharedProbeStatus : std::uint8_t {
  Failed,
  SharedVerified,
  NoSharedCapabilityVerified,
};

struct RuntimeSharedProbe final {
  RuntimeSharedProbeStatus status{RuntimeSharedProbeStatus::Failed};
  std::optional<rund::node::accel::detail::RangeGpuShape> shape{};
};

[[nodiscard]] constexpr std::optional<rund::node::accel::detail::RangeGpuShape>
RangeExecShape(const rund::node::accel::detail::RangePlan &plan) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeExec> execution = RangeExec::from(plan);
  return execution.has_value()
             ? std::optional<RangeGpuShape>{execution->shape()}
             : std::nullopt;
}

#if defined(__APPLE__)
[[nodiscard]] RuntimeSharedProbe
MetalMaximumSharedShapeContract(const rund::AccelDevice &pick) {
  using namespace rund::node::accel::detail;
  std::array<rund::kernel::u32, 515u> input{};
  auto fixture = stencil::match::BuildResources(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, 1u, input);
  if (!fixture.context.check.ok || !fixture.input.check.ok ||
      !fixture.output.check.ok) {
    std::cerr << "metal stencil immutable fixture unavailable: context="
              << fixture.context.check.reason
              << " input=" << fixture.input.check.reason
              << " output=" << fixture.output.check.reason << '\n';
    return {};
  }
  const ContextAdmission admission = AdmitContextForSupport(fixture.context);
  std::shared_ptr<void> input_handle;
  std::shared_ptr<void> output_handle;
  if (!admission.check.ok || admission.pick == nullptr ||
      !ValidateAccelBufferForSupport(admission, fixture.input, input_handle)
           .ok ||
      !ValidateAccelBufferForSupport(admission, fixture.output, output_handle)
           .ok) {
    return {};
  }
  const RangeBinds bindings{
      .input = &fixture.input.resident,
      .input_handle = &input_handle,
      .output = &fixture.output.resident,
      .output_handle = &output_handle,
  };
  std::optional<RangeGpuShape> maximum{};
  {
    const KernelPreparationScope preparation{
        KernelPreparationMode::PipelinePrivate};
    std::shared_ptr<void> first;
    rund::kernel::StencilDesc selected_desc{};
    rund::kernel::StencilPlan selected_plan{};
    for (rund::kernel::u64 radius = 256u; radius != 0u; --radius) {
      const rund::kernel::StencilDesc desc{
          .op = rund::kernel::StencilOp::Sum,
          .element = rund::kernel::StencilElement::U32,
          .boundary = rund::kernel::StencilBoundary::Clamp,
          .element_count = input.size(),
          .radius = radius,
      };
      const rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
      const RangePlan range = MetalStencilRangePlan(
          admission.pick->raw, plan, rund::kernel::ComputeDomain::U32);
      std::shared_ptr<void> candidate;
      const rund::AccelCheck check = PrepareMetalStencil(
          admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
          bindings, range, candidate);
      const auto *const native =
          static_cast<const MetalRangeResources *>(candidate.get());
      const std::optional<RangeGpuShape> native_shape =
          native == nullptr ? std::nullopt : RangeExecShape(native->range);
      if (!plan.ok || !range.ok() || !check.ok || native == nullptr ||
          !native_shape.has_value() ||
          native->stage_count != range.stage_count() ||
          native->stage_count == 0u || native->pipelines[0u] == nullptr) {
        std::cerr << "metal maximum shared probe failed: radius=" << radius
                  << " check=" << check.ok << " reason=" << check.reason
                  << " native=" << (native != nullptr) << '\n';
        return {};
      }
      if (!native_shape->uses_shared_halo()) {
        continue;
      }
      if (native_shape->shared_radius_capacity() != radius ||
          input.size() % native_shape->width() != 3u ||
          RangePhysicalGroupCount(input.size(), *native_shape) <= 1u) {
        std::cerr << "metal maximum shared shape mismatch: radius=" << radius
                  << " width=" << native_shape->width()
                  << " cap=" << native_shape->shared_radius_capacity() << '\n';
        return {};
      }
      maximum = *native_shape;
      selected_desc = desc;
      selected_plan = plan;
      first = std::move(candidate);
      break;
    }
    if (!maximum.has_value()) {
      return {RuntimeSharedProbeStatus::NoSharedCapabilityVerified, {}};
    }

    const auto *const first_native =
        static_cast<const MetalRangeResources *>(first.get());
    MetalKernelImmutablePipelines immutable{};
    if (first_native->stage_count != 1u ||
        first_native->pipelines[0u] == nullptr) {
      return {};
    }
    immutable.stages[0u] = first_native->pipelines[0u];
    immutable.count = 1u;
    rund::node::accel::ResetRuntimeStats(fixture.context.pick);
    std::shared_ptr<void> second;
    const RangePlan selected_range = MetalStencilRangePlan(
        admission.pick->raw, selected_plan, rund::kernel::ComputeDomain::U32);
    const rund::AccelCheck second_check =
        PrepareMetalStencil(admission.pick->raw, selected_desc, selected_plan,
                            rund::kernel::ComputeDomain::U32, bindings,
                            selected_range, second, &immutable);
    const auto *const second_native =
        static_cast<const MetalRangeResources *>(second.get());
    const std::optional<RangeGpuShape> second_shape =
        second_native == nullptr ? std::nullopt
                                 : RangeExecShape(second_native->range);
    const rund::RuntimeStats stats =
        rund::node::accel::ReadRuntimeStats(fixture.context.pick);
    const bool matched =
        second_check.ok && second_native != nullptr &&
        second_shape.has_value() && second_shape->uses_shared_halo() &&
        *second_shape == *maximum && stats.ok &&
        stats.pipeline_compile_count == 0u &&
        stats.pipeline_cache_hit_count == 0u &&
        second_native->stage_count == 1u &&
        second_native->pipelines[0u] == first_native->pipelines[0u];
    if (!matched) {
      std::cerr << "metal maximum shared immutable borrow mismatch: check="
                << second_check.ok << " reason=" << second_check.reason
                << " native=" << (second_native != nullptr)
                << " compile=" << stats.pipeline_compile_count
                << " hit=" << stats.pipeline_cache_hit_count << '\n';
      return {};
    }
  }
  if (!stencil::MatchesCapabilitySharedBoundaryU32(
          pick, maximum->shared_radius_capacity())) {
    return {};
  }
  return {RuntimeSharedProbeStatus::SharedVerified, maximum};
}
#endif

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
[[nodiscard]] RuntimeSharedProbe VulkanMaximumSharedShapeContract(
    const rund::AccelDevice &pick,
    const rund::node::accel::detail::VulkanAdapter &adapter) {
  using namespace rund::node::accel::detail;
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  if (token == nullptr || token->raw.backend.context != &adapter) {
    std::cerr << "vulkan maximum shared raw pick unavailable\n";
    return {};
  }
  std::optional<RangeGpuShape> maximum{};
  for (rund::kernel::u64 radius = 256u; radius != 0u; --radius) {
    const rund::kernel::StencilPlan plan =
        rund::kernel::PlanStencil(rund::kernel::StencilDesc{
            .op = rund::kernel::StencilOp::Sum,
            .element = rund::kernel::StencilElement::U32,
            .boundary = rund::kernel::StencilBoundary::Clamp,
            .element_count = 515u,
            .radius = radius});
    const RangePlan range = VulkanStencilRangePlan(
        token->raw, plan, rund::kernel::ComputeDomain::U32);
    const std::optional<RangeGpuShape> shape = RangeGpuShapeFor(range);
    if (!shape.has_value()) {
      std::cerr << "vulkan maximum shared probe invalid: radius=" << radius
                << '\n';
      return {};
    }
    if (!shape->uses_shared_halo()) {
      continue;
    }
    if (shape->shared_radius_capacity() != radius ||
        515u % shape->width() != 3u ||
        RangePhysicalGroupCount(515u, *shape) <= 1u) {
      std::cerr << "vulkan maximum shared shape mismatch: radius=" << radius
                << " width=" << shape->width()
                << " cap=" << shape->shared_radius_capacity() << '\n';
      return {};
    }
    maximum = *shape;
    break;
  }
  if (!maximum.has_value()) {
    return {RuntimeSharedProbeStatus::NoSharedCapabilityVerified, {}};
  }

  std::array<rund::kernel::u32, 515u> input{};
  auto fixture = stencil::match::BuildResources(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, maximum->shared_radius_capacity(),
      input);
  if (!fixture.context.check.ok || !fixture.input.check.ok ||
      !fixture.output.check.ok) {
    return {};
  }
  const ContextAdmission admission = AdmitContextForSupport(fixture.context);
  std::shared_ptr<void> input_handle;
  std::shared_ptr<void> output_handle;
  if (!admission.check.ok || admission.pick == nullptr ||
      !ValidateAccelBufferForSupport(admission, fixture.input, input_handle)
           .ok ||
      !ValidateAccelBufferForSupport(admission, fixture.output, output_handle)
           .ok) {
    return {};
  }
  const rund::kernel::StencilDesc desc{
      .op = rund::kernel::StencilOp::Sum,
      .element = rund::kernel::StencilElement::U32,
      .boundary = rund::kernel::StencilBoundary::Clamp,
      .element_count = input.size(),
      .radius = maximum->shared_radius_capacity(),
  };
  const rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  const RangeBinds bindings{
      .input = &fixture.input.resident,
      .input_handle = &input_handle,
      .output = &fixture.output.resident,
      .output_handle = &output_handle,
  };
  {
    const KernelPreparationScope preparation{
        KernelPreparationMode::PipelinePrivate};
    std::shared_ptr<void> first;
    const RangePlan range = VulkanStencilRangePlan(
        admission.pick->raw, plan, rund::kernel::ComputeDomain::U32);
    const rund::AccelCheck first_check = PrepareVulkanStencil(
        admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
        bindings, range, first);
    const auto *const first_native =
        static_cast<const VulkanRangeResources *>(first.get());
    const std::optional<RangeGpuShape> first_shape =
        first_native == nullptr ? std::nullopt
                                : RangeExecShape(first_native->range);
    if (!plan.ok || !range.ok() || !first_check.ok || first_native == nullptr ||
        first_native->stage_count != range.stage_count() ||
        first_native->stage_count != 1u ||
        first_native->pipelines[0u] == nullptr || !first_shape.has_value() ||
        !first_shape->uses_shared_halo() || *first_shape != *maximum) {
      return {};
    }
    VulkanKernelImmutablePipelines immutable{};
    immutable.kind = rund::kernel::NodeKind::Stencil;
    if (!immutable.append(first_native->pipelines[0u],
                          RangeDescriptorCount(range), 1u)) {
      return {};
    }
    rund::node::accel::ResetRuntimeStats(fixture.context.pick);
    std::shared_ptr<void> second;
    const rund::AccelCheck second_check = PrepareVulkanStencil(
        admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
        bindings, range, second, &immutable);
    const auto *const second_native =
        static_cast<const VulkanRangeResources *>(second.get());
    const std::optional<RangeGpuShape> second_shape =
        second_native == nullptr ? std::nullopt
                                 : RangeExecShape(second_native->range);
    const rund::RuntimeStats stats =
        rund::node::accel::ReadRuntimeStats(fixture.context.pick);
    if (!second_check.ok || second_native == nullptr ||
        !second_shape.has_value() || !second_shape->uses_shared_halo() ||
        *second_shape != *maximum || second_native->stage_count != 1u ||
        second_native->pipelines[0u] != first_native->pipelines[0u] ||
        !stats.ok || stats.pipeline_compile_count != 0u ||
        stats.pipeline_cache_hit_count != 0u) {
      return {};
    }
  }
  if (!stencil::MatchesCapabilitySharedBoundaryU32(
          pick, maximum->shared_radius_capacity())) {
    return {};
  }
  return {RuntimeSharedProbeStatus::SharedVerified, maximum};
}
#endif

[[nodiscard]] bool StencilSourcesCarryConvergentSharedHalo() {
  using namespace rund::node::accel::detail;
  for (const rund::kernel::u32 width : kRangeWidths) {
    const RangeGpuShape requested = stencil::RangeSharedShape(width, width);
    const RangePlan metal_range = stencil::PlanStencilSourceVariant(
        RangeSource::Metal, rund::kernel::StencilOp::Sum,
        rund::kernel::ComputeDomain::U64, requested,
        stencil::SourcePlanPath::SharedHalo);
    const RangeGpuShape shape = stencil::RequireRangeShape(metal_range);
    if (!metal_range.ok() || shape != requested) {
      return false;
    }
    const std::string metal =
        MetalRangeSource(stencil::RequireRangeExec(metal_range));
    const std::string metal_tile =
        "threadgroup uint tile[" + std::to_string(3u * width) + "];";
    const std::size_t metal_center =
        metal.find("const uint center_value = input[");
    const std::size_t metal_left_fan =
        metal.find("left_inputs == 0u && tid == 0u", metal_center);
    const std::size_t metal_right_fan =
        metal.find("right_inputs == 0u && ulong(tid) + 1ul == active_lanes",
                   metal_left_fan);
    const std::size_t metal_barrier = metal.find(
        "threadgroup_barrier(mem_flags::mem_threadgroup);", metal_right_fan);
    const std::size_t metal_guard = metal.find(
        "if (ulong(tid) >= active_lanes) { return; }", metal_barrier);
    if (metal.find(metal_tile) == std::string::npos ||
        metal.find("* " + std::to_string(width) + "ul;") == std::string::npos ||
        metal_center == std::string::npos ||
        metal_left_fan == std::string::npos ||
        metal_right_fan == std::string::npos ||
        metal_barrier == std::string::npos ||
        metal_guard == std::string::npos ||
        metal.find("for (ulong step = 1ul;") != std::string::npos) {
      return false;
    }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
    const RangePlan vulkan_range = stencil::PlanStencilSourceVariant(
        RangeSource::Vulkan, rund::kernel::StencilOp::Sum,
        rund::kernel::ComputeDomain::U64, requested,
        stencil::SourcePlanPath::SharedHalo);
    const RangeGpuShape vulkan_shape = stencil::RequireRangeShape(vulkan_range);
    if (!vulkan_range.ok() || vulkan_shape != requested) {
      return false;
    }
    const std::string vulkan =
        VulkanRangeSource(stencil::RequireRangeExec(vulkan_range));
    const std::string vulkan_group =
        "layout(local_size_x = " + std::to_string(width) + ") in;";
    const std::string vulkan_tile =
        "shared uint64_t range_tile[" + std::to_string(3u * width) + "];";
    const std::size_t vulkan_center =
        vulkan.find("const uint64_t center_value = input_values[");
    const std::size_t vulkan_left_fan =
        vulkan.find("left_inputs == 0u && lane == 0u", vulkan_center);
    const std::size_t vulkan_right_fan = vulkan.find(
        "right_inputs == 0u && uint64_t(lane) + uint64_t(1) == active_lanes",
        vulkan_left_fan);
    const std::size_t vulkan_barrier =
        vulkan.find("barrier();", vulkan_right_fan);
    const std::size_t vulkan_guard = vulkan.find(
        "if (uint64_t(lane) >= active_lanes) { return; }", vulkan_barrier);
    const RangePlan vulkan_u32_range = stencil::PlanStencilSourceVariant(
        RangeSource::Vulkan, rund::kernel::StencilOp::Sum,
        rund::kernel::ComputeDomain::U32, requested,
        stencil::SourcePlanPath::SharedHalo);
    const std::string vulkan_u32 =
        VulkanRangeSource(stencil::RequireRangeExec(vulkan_u32_range));
    const std::string vulkan_extended = vulkan + '\n';
    std::uint64_t vulkan_upper = 0u;
    if (!VulkanRangeSourceBytes(stencil::RequireRangeExec(vulkan_range),
                                vulkan_upper)) {
      return false;
    }
    if (vulkan.find(vulkan_group) == std::string::npos ||
        vulkan.find(vulkan_tile) == std::string::npos ||
        vulkan_center == std::string::npos ||
        vulkan_left_fan == std::string::npos ||
        vulkan_right_fan == std::string::npos ||
        vulkan_barrier == std::string::npos ||
        vulkan_guard == std::string::npos ||
        vulkan.find("for (uint64_t step = uint64_t(1);") != std::string::npos ||
        vulkan_u32.find("uint value = range_tile[first];") ==
            std::string::npos ||
        vulkan_u32.find("uint64_t value") != std::string::npos ||
        vulkan_u32.find("uint64_t(range_tile[center - step])") !=
            std::string::npos ||
        !VulkanRangeSourceMatches(stencil::RequireRangeExec(vulkan_range),
                                  vulkan, SourceHash(vulkan)) ||
        VulkanRangeSourceMatches(stencil::RequireRangeExec(vulkan_range),
                                 vulkan_extended,
                                 SourceHash(vulkan_extended)) ||
        VulkanRangeSourceMatches(stencil::RequireRangeExec(vulkan_range),
                                 vulkan, SourceHash(vulkan) ^ 1u) ||
        vulkan.size() != vulkan_upper || vulkan_u32.empty()) {
      return false;
    }
#endif
  }

  const RangeGpuShape direct_requested = stencil::RangeDirectShape(64u);
  const RangePlan metal_direct_range = stencil::PlanStencilSourceVariant(
      RangeSource::Metal, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, direct_requested,
      stencil::SourcePlanPath::Direct);
  const RangeGpuShape direct = stencil::RequireRangeShape(metal_direct_range);
  if (!metal_direct_range.ok() || direct != direct_requested) {
    return false;
  }
  const std::string metal_direct =
      MetalRangeSource(stencil::RequireRangeExec(metal_direct_range));
  if (metal_direct.find("threadgroup uint tile[") != std::string::npos ||
      metal_direct.find("threadgroup_barrier") != std::string::npos ||
      metal_direct.find(
          "for (ulong slot = 0ul; slot < params.window_size; ++slot)") ==
          std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const RangePlan vulkan_direct_range = stencil::PlanStencilSourceVariant(
      RangeSource::Vulkan, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, direct_requested,
      stencil::SourcePlanPath::Direct);
  if (!vulkan_direct_range.ok()) {
    return false;
  }
  const std::string vulkan_direct =
      VulkanRangeSource(stencil::RequireRangeExec(vulkan_direct_range));
  std::uint64_t direct_upper = 0u;
  if (!VulkanRangeSourceBytes(stencil::RequireRangeExec(vulkan_direct_range),
                              direct_upper)) {
    return false;
  }
  if (vulkan_direct.find("shared uint range_tile[") != std::string::npos ||
      vulkan_direct.find("barrier();") != std::string::npos ||
      vulkan_direct.find(
          "for (uint64_t slot = uint64_t(0); slot < params.window_size;") ==
          std::string::npos ||
      vulkan_direct.size() != direct_upper) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] bool RangeSourcesCarryLinearFamilies() {
  using namespace rund::node::accel::detail;
  constexpr RangeGpuShape physical = stencil::RangeDirectShape(64u);
  constexpr RangePlan metal_prefix = stencil::PlanStencilSourceVariant(
      RangeSource::Metal, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, physical,
      stencil::SourcePlanPath::PrefixDifference);
  constexpr RangePlan metal_block = stencil::PlanStencilSourceVariant(
      RangeSource::Metal, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, physical,
      stencil::SourcePlanPath::BlockPrefixSuffix);
  constexpr RangePlan metal_direct = stencil::PlanStencilSourceVariant(
      RangeSource::Metal, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, physical,
      stencil::SourcePlanPath::Direct);
  static_assert(metal_prefix.ok() && metal_block.ok() && metal_direct.ok());
  static_assert(RangeStaticSharedBytes(metal_prefix) == 64u * 4u);
  static_assert(RangeStaticSharedBytes(metal_block) == 0u);
  if (metal_prefix.source_identity() == metal_direct.source_identity() ||
      metal_prefix.source_identity() == metal_block.source_identity() ||
      metal_block.source_identity() == metal_direct.source_identity()) {
    return false;
  }
  const std::string metal_prefix_source =
      MetalRangeSource(stencil::RequireRangeExec(metal_prefix));
  const std::string metal_block_source =
      MetalRangeSource(stencil::RequireRangeExec(metal_block));
  std::uint64_t metal_prefix_upper = 0u;
  std::uint64_t metal_block_upper = 0u;
  if (!MetalRangeSourceUpperBytes(stencil::RequireRangeExec(metal_prefix),
                                  metal_prefix_upper) ||
      !MetalRangeSourceUpperBytes(stencil::RequireRangeExec(metal_block),
                                  metal_block_upper) ||
      metal_prefix_source.size() != metal_prefix_upper ||
      metal_block_source.size() != metal_block_upper ||
      metal_prefix_source.find("device uint* scratch0 [[buffer(3)]],") ==
          std::string::npos ||
      metal_prefix_source.find("threadgroup uint scan[64];") ==
          std::string::npos ||
      metal_prefix_source.find("scratch0[i] = scan[tid] + value;") ==
          std::string::npos ||
      metal_prefix_source.find("value -= scratch0[left - 1ul];") ==
          std::string::npos ||
      metal_block_source.find("device int* forward_values [[buffer(3)]],") ==
          std::string::npos ||
      metal_block_source.find("const ulong window = params.window_size;") ==
          std::string::npos ||
      metal_block_source.find("backward_values[index]") == std::string::npos ||
      metal_block_source.find("output[i] = min(backward_values[left], "
                              "forward_values[right]);") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  constexpr RangePlan vulkan_prefix = stencil::PlanStencilSourceVariant(
      RangeSource::Vulkan, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, physical,
      stencil::SourcePlanPath::PrefixDifference);
  constexpr RangePlan vulkan_block = stencil::PlanStencilSourceVariant(
      RangeSource::Vulkan, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, physical,
      stencil::SourcePlanPath::BlockPrefixSuffix);
  static_assert(vulkan_prefix.ok() && vulkan_block.ok());
  const std::string vulkan_prefix_source =
      VulkanRangeSource(stencil::RequireRangeExec(vulkan_prefix));
  const std::string vulkan_block_source =
      VulkanRangeSource(stencil::RequireRangeExec(vulkan_block));
  std::uint64_t vulkan_prefix_upper = 0u;
  std::uint64_t vulkan_block_upper = 0u;
  if (!VulkanRangeSourceBytes(stencil::RequireRangeExec(vulkan_prefix),
                              vulkan_prefix_upper) ||
      !VulkanRangeSourceBytes(stencil::RequireRangeExec(vulkan_block),
                              vulkan_block_upper) ||
      vulkan_prefix_source.size() != vulkan_prefix_upper ||
      vulkan_block_source.size() != vulkan_block_upper ||
      vulkan_prefix_source.find("shared uint range_scan[64];") ==
          std::string::npos ||
      vulkan_prefix_source.find(
          "scratch0_values[uint(index)] = range_scan[lane] + value;") ==
          std::string::npos ||
      vulkan_prefix_source.find(
          "value -= scratch0_values[uint(left - uint64_t(1))];") ==
          std::string::npos ||
      vulkan_block_source.find("#define value_type int") == std::string::npos ||
      vulkan_block_source.find("const uint64_t window = params.window_size;") ==
          std::string::npos ||
      vulkan_block_source.find("scratch1_values[uint(left)]") ==
          std::string::npos ||
      !VulkanRangeSourceMatches(stencil::RequireRangeExec(vulkan_prefix),
                                vulkan_prefix_source,
                                SourceHash(vulkan_prefix_source)) ||
      !VulkanRangeSourceMatches(stencil::RequireRangeExec(vulkan_block),
                                vulkan_block_source,
                                SourceHash(vulkan_block_source))) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] bool BackendRunsStencilShapeCases(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") &&
         StencilMatch(stencil::MatchesCount65U32(pick), "sum.u32.count65") &&
         StencilMatch(stencil::MatchesRadius64BoundaryU32(pick),
                      "sum.u32.count259.radius64") &&
         StencilMatch(stencil::MatchesPrefixDifferenceU32(pick),
                      "sum.u32.prefix-difference");
}

[[nodiscard]] bool
RuntimeSharedProbeMatchesContract(const RuntimeSharedProbe probe,
                                  const char *const backend) {
  switch (probe.status) {
  case RuntimeSharedProbeStatus::SharedVerified:
    if (probe.shape.has_value() && probe.shape->uses_shared_halo() &&
        probe.shape->shared_radius_capacity() != 0u) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::NoSharedCapabilityVerified:
    if (!probe.shape.has_value()) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::Failed:
    break;
  }
  std::cerr << backend << " runtime maximum shared contract failed: status="
            << static_cast<unsigned>(probe.status) << " width="
            << (probe.shape.has_value() ? probe.shape->width() : 0u) << " cap="
            << (probe.shape.has_value() ? probe.shape->shared_radius_capacity()
                                        : 0u)
            << '\n';
  return false;
}

[[nodiscard]] bool BackendRunsStencilValueCases(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesU64(pick), "sum.u64") &&
         StencilMatch(stencil::MatchesMinU32(pick), "min.u32") &&
         StencilMatch(stencil::MatchesMinI32(pick), "min.i32") &&
         StencilMatch(stencil::MatchesMaxU64(pick), "max.u64") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMinI32(pick),
                      "min.i32.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMaxI32(pick),
                      "max.i32.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMinU64(pick),
                      "min.u64.block-prefix-suffix") &&
         StencilMatch(stencil::MatchesBlockPrefixSuffixMaxU64(pick),
                      "max.u64.block-prefix-suffix");
}

[[nodiscard]] bool BackendRunsRangeParity(const rund::AccelDevice &pick) {
  return StencilMatch(stencil::MatchesForcedPrefixDifferenceU32(pick),
                      "sum.u32.direct-prefix-difference") &&
         StencilMatch(stencil::MatchesForcedPrefixDifferenceU64(pick),
                      "sum.u64.direct-prefix-difference") &&
         StencilMatch(stencil::MatchesDeepPrefixHierarchyU32(pick),
                      "sum.u32.prefix-hierarchy") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMinI32(pick),
                      "min.i32.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMaxI32(pick),
                      "max.i32.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMinU64(pick),
                      "min.u64.direct-block-prefix-suffix") &&
         StencilMatch(stencil::MatchesForcedBlockPrefixSuffixMaxU64(pick),
                      "max.u64.direct-block-prefix-suffix");
}

[[nodiscard]] bool BackendRunsStencilRemainder(const rund::AccelDevice &pick) {
  return BackendRunsStencilShapeCases(pick) &&
         BackendRunsStencilValueCases(pick);
}

class MetalStencilVariantCacheContract final {
public:
  explicit MetalStencilVariantCacheContract(
      rund::node::accel::detail::MetalAdapter &adapter)
      : adapter_(adapter) {
    valid_ = Snapshot();
  }

  [[nodiscard]] bool Observe(const rund::AccelDevice &pick,
                             const char *const name) {
    const std::size_t before = seen_count_;
    if (!valid_ || !Snapshot()) {
      return false;
    }
    const std::size_t inserted = seen_count_ - before;
    const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
    const bool cold = inserted != 0u;
    const bool matched =
        stats.ok && (cold ? stats.pipeline_compile_count >= inserted
                          : stats.pipeline_cache_hit_count != 0u);
    if (!matched) {
      std::cerr << "metal stencil capability-derived identity mismatch: "
                << name << " new_variants=" << inserted
                << " compile=" << stats.pipeline_compile_count
                << " hit=" << stats.pipeline_cache_hit_count << '\n';
      return false;
    }
    observed_cold_ = observed_cold_ || cold;
    observed_hit_ = observed_hit_ || !cold;
    return true;
  }

  [[nodiscard]] bool complete() const noexcept {
    return valid_ && observed_cold_ && observed_hit_;
  }

private:
  [[nodiscard]] static bool Tracked(const std::string &name) noexcept {
    return name.find("range.aggregate.") != std::string::npos;
  }

  [[nodiscard]] bool Seen(const std::string &name) const noexcept {
    return std::find(seen_.begin(), seen_.begin() + seen_count_, name) !=
           seen_.begin() + seen_count_;
  }

  [[nodiscard]] bool Snapshot() {
    std::lock_guard<std::mutex> lock{adapter_.mutex};
    for (const rund::node::accel::detail::MetalNamedPipeline &pipeline :
         adapter_.named_pipelines) {
      if (!Tracked(pipeline.name) || Seen(pipeline.name)) {
        continue;
      }
      if (seen_count_ == seen_.size()) {
        valid_ = false;
        return false;
      }
      seen_[seen_count_++] = pipeline.name;
    }
    return true;
  }

  rund::node::accel::detail::MetalAdapter &adapter_;
  std::array<std::string, 16u> seen_{};
  std::size_t seen_count_{};
  bool valid_{};
  bool observed_cold_{};
  bool observed_hit_{};
};

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
class VulkanStencilVariantCacheContract final {
public:
  explicit VulkanStencilVariantCacheContract(
      const rund::node::accel::detail::VulkanAdapter &) noexcept {}

  [[nodiscard]] bool Observe(const rund::AccelDevice &pick,
                             const rund::kernel::u64 element_count,
                             const rund::kernel::u64 radius,
                             const rund::kernel::ComputeDomain domain,
                             const char *const name) noexcept {
    using namespace rund::node::accel::detail;
    const rund::kernel::StencilPlan plan =
        rund::kernel::PlanStencil(rund::kernel::StencilDesc{
            .op = rund::kernel::StencilOp::Sum,
            .element = rund::kernel::StencilElement::U32,
            .boundary = rund::kernel::StencilBoundary::Clamp,
            .element_count = element_count,
            .radius = radius});
    const std::shared_ptr<PickToken> token = AdmitPick(pick);
    const RangePlan range =
        token != nullptr ? VulkanStencilRangePlan(token->raw, plan, domain)
                         : RangePlan::rejected("compute_adapter_unavailable");
    const std::optional<RangeGpuShape> shape = RangeGpuShapeFor(range);
    if (!shape.has_value()) {
      std::cerr << "vulkan stencil variant unavailable: " << name
                << " range=" << range.reason() << '\n';
      return false;
    }
    const RangeIdentity identity = range.source_identity();
    const bool expected_hit =
        std::find(seen_.begin(), seen_.begin() + seen_count_, identity) !=
        seen_.begin() + seen_count_;
    const std::uint64_t stages = range.stage_count();
    const rund::RuntimeStats stats = rund::node::accel::ReadRuntimeStats(pick);
    const bool matched =
        stats.ok && stages != 0u &&
        stats.pipeline_compile_count == (expected_hit ? 0u : 1u) &&
        stats.pipeline_cache_hit_count == (expected_hit ? stages : stages - 1u);
    if (!matched) {
      std::cerr << "vulkan stencil capability-derived identity mismatch: "
                << name << " width=" << shape->width()
                << " radius_cap=" << shape->shared_radius_capacity()
                << " stages=" << stages
                << " expected=" << (expected_hit ? "hit" : "cold")
                << " compile=" << stats.pipeline_compile_count
                << " hit=" << stats.pipeline_cache_hit_count << '\n';
      return false;
    }
    if (!expected_hit) {
      if (seen_count_ == seen_.size()) {
        return false;
      }
      seen_[seen_count_++] = identity;
    }
    return true;
  }

private:
  std::array<rund::node::accel::detail::RangeIdentity, 8u> seen_{};
  std::size_t seen_count_{};
};
#endif

} // namespace

bool BackendRunsStencil(const rund::AccelDevice &pick) {
  return StencilMatch(RangeDispatchIsExact(), "shape.dispatch") &&
         StencilMatch(RangePlanProjectionIsExact(), "shape.projection") &&
         StencilMatch(MetalStencilRejectedCompileTelemetryIsExact(),
                      "metal.fallback-telemetry") &&
         StencilMatch(MetalStencilNamedPipelinePublicationIsTransactional(),
                      "metal.pipeline-publication") &&
         StencilMatch(MetalRangeSourcePublicationIsTransactional(),
                      "metal.source-publication") &&
         StencilMatch(MetalRangeSourceRetryIsExact(), "metal.source-retry") &&
         StencilMatch(StencilShapeRejectsOnlyOverlappingStorage(),
                      "shape.storage") &&
         StencilMatch(StencilSourcesCarryConvergentSharedHalo(),
                      "source.variants") &&
         StencilMatch(RangeSourcesCarryLinearFamilies(),
                      "source.linear-families") &&
         StencilMatch(SignedStencilSourcesCarryDomainOrder(),
                      "source.domain") &&
         StencilMatch(stencil::MatchesU32(pick), "sum.u32") &&
         BackendRunsStencilRemainder(pick);
}

bool RequiredMetalRunsStencil() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  if (!StencilMatch(pick.api == rund::AccelApi::Metal, "pick.api") ||
      !StencilMatch(RangeDispatchIsExact(), "shape.dispatch") ||
      !StencilMatch(RangePlanProjectionIsExact(), "shape.projection") ||
      !StencilMatch(MetalStencilRejectedCompileTelemetryIsExact(),
                    "metal.fallback-telemetry") ||
      !StencilMatch(MetalStencilNamedPipelinePublicationIsTransactional(),
                    "metal.pipeline-publication") ||
      !StencilMatch(MetalRangeSourcePublicationIsTransactional(),
                    "metal.source-publication") ||
      !StencilMatch(MetalRangeSourceRetryIsExact(), "metal.source-retry") ||
      !StencilMatch(StencilShapeRejectsOnlyOverlappingStorage(),
                    "shape.storage") ||
      !StencilMatch(StencilSourcesCarryConvergentSharedHalo(),
                    "source.variants") ||
      !StencilMatch(RangeSourcesCarryLinearFamilies(),
                    "source.linear-families") ||
      !StencilMatch(SignedStencilSourcesCarryDomainOrder(), "source.domain")) {
    return false;
  }
  auto *const adapter = static_cast<rund::node::accel::detail::MetalAdapter *>(
      pick.backend.context);
  if (adapter == nullptr || adapter->device == nullptr) {
    std::cerr << "metal stencil adapter unavailable after successful pick\n";
    return false;
  }
  MetalStencilVariantCacheContract variants{*adapter};
  if (!StencilMatch(stencil::MatchesU32(pick), "sum.u32") ||
      !variants.Observe(pick, "count6.radius1") ||
      !StencilMatch(stencil::MatchesSumI32(pick), "sum.i32") ||
      !variants.Observe(pick, "count6.radius1.signed-sum") ||
      !StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") ||
      !variants.Observe(pick, "count6.radius2") ||
      !StencilMatch(stencil::MatchesCount65U32(pick), "sum.u32.count65") ||
      !variants.Observe(pick, "count65.radius1") ||
      !StencilMatch(stencil::MatchesRadius64BoundaryU32(pick),
                    "sum.u32.count259.radius64") ||
      !variants.Observe(pick, "count259.radius64") ||
      !StencilMatch(stencil::MatchesPrefixDifferenceU32(pick),
                    "sum.u32.prefix-difference") ||
      !variants.Observe(pick, "count257.prefix") || !variants.complete()) {
    return false;
  }
#if defined(__APPLE__)
  return BackendRunsStencilValueCases(pick) && BackendRunsRangeParity(pick) &&
         StencilMatch(RuntimeSharedProbeMatchesContract(
                          MetalMaximumSharedShapeContract(pick), "metal"),
                      "metal.maximum-shared-capability");
#else
  return false;
#endif
}

bool RequiredVulkanRunsStencil() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  if (!StencilMatch(pick.api == rund::AccelApi::Vulkan, "pick.api")) {
    return false;
  }
  const auto *const adapter =
      static_cast<const rund::node::accel::detail::VulkanAdapter *>(
          pick.backend.context);
  if (adapter == nullptr || adapter->device == VK_NULL_HANDLE ||
      adapter->physical_device == VK_NULL_HANDLE) {
    std::cerr << "vulkan stencil adapter unavailable after successful pick\n";
    return false;
  }
  VulkanStencilVariantCacheContract variants{*adapter};
  if (!StencilMatch(StencilSourcesCarryConvergentSharedHalo(),
                    "source.variants") ||
      !StencilMatch(RangeSourcesCarryLinearFamilies(),
                    "source.linear-families") ||
      !StencilMatch(SignedStencilSourcesCarryDomainOrder(), "source.domain") ||
      !StencilMatch(stencil::MatchesU32(pick), "sum.u32") ||
      !variants.Observe(pick, 6u, 1u, rund::kernel::ComputeDomain::U32,
                        "count6.radius1")) {
    return false;
  }
  if (!StencilMatch(stencil::MatchesSumI32(pick), "sum.i32") ||
      !variants.Observe(pick, 6u, 1u, rund::kernel::ComputeDomain::I32,
                        "count6.radius1.signed-sum") ||
      !StencilMatch(stencil::MatchesWideWindowU32(pick), "sum.u32.radius2") ||
      !variants.Observe(pick, 6u, 2u, rund::kernel::ComputeDomain::U32,
                        "count6.radius2") ||
      !StencilMatch(stencil::MatchesCount65U32(pick), "sum.u32.count65") ||
      !variants.Observe(pick, 65u, 1u, rund::kernel::ComputeDomain::U32,
                        "count65.radius1") ||
      !StencilMatch(stencil::MatchesRadius64BoundaryU32(pick),
                    "sum.u32.count259.radius64") ||
      !variants.Observe(pick, 259u, 64u, rund::kernel::ComputeDomain::U32,
                        "count259.radius64") ||
      !StencilMatch(stencil::MatchesPrefixDifferenceU32(pick),
                    "sum.u32.prefix-difference") ||
      !variants.Observe(pick, 257u, 257u, rund::kernel::ComputeDomain::U32,
                        "count257.prefix")) {
    return false;
  }
  return BackendRunsStencilValueCases(pick) && BackendRunsRangeParity(pick) &&
         StencilMatch(
             RuntimeSharedProbeMatchesContract(
                 VulkanMaximumSharedShapeContract(pick, *adapter), "vulkan"),
             "vulkan.maximum-shared-capability");
#else
  return false;
#endif
}

} // namespace node_accel_contract
