#include <accel/api.hpp>
#include <accel/device.hpp>
#include <accel/runtime.hpp>

#include "src/accel/context/internal/support.hpp"
#include "src/accel/kernel/preparation.hpp"
#include "src/accel/metal/pipeline/cache.hpp"
#include "src/accel/metal/pipeline/template.hpp"
#include "src/accel/metal/stencil/local.hpp"
#include "src/accel/range_aggregate/plan.hpp"
#include "src/accel/source/hash.hpp"
#include "src/accel/stencil/shape.hpp"
#include "src/accel/vulkan/kernel/pipeline/template.hpp"
#include "src/accel/vulkan/stencil/local.hpp"
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

[[nodiscard]] rund::node::accel::detail::RangeAggregatePlan
MetalStencilRangePlan(const rund::AccelDevice &pick,
                      const rund::kernel::StencilPlan &plan,
                      const rund::kernel::ComputeDomain domain) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeAggregateShape> shape =
      RangeAggregateShape::from_stencil(plan, domain);
  return shape.has_value()
             ? PlanRangeAggregate(*shape, MetalRangeAggregateCapabilities(pick))
             : RangeAggregatePlan::rejected("accel_kernel_graph_invalid");
}

[[nodiscard, maybe_unused]] rund::node::accel::detail::RangeAggregatePlan
VulkanStencilRangePlan(const rund::AccelDevice &pick,
                       const rund::kernel::StencilPlan &plan,
                       const rund::kernel::ComputeDomain domain) noexcept {
  using namespace rund::node::accel::detail;
  const std::optional<RangeAggregateShape> shape =
      RangeAggregateShape::from_stencil(plan, domain);
  return shape.has_value()
             ? PlanRangeAggregate(*shape,
                                  VulkanRangeAggregateCapabilities(pick))
             : RangeAggregatePlan::rejected("accel_kernel_graph_invalid");
}

[[nodiscard]] bool SignedStencilSourcesCarryDomainOrder() {
  using namespace rund::node::accel::detail;
  constexpr StencilGpuShape requested = StencilGpuShape::shared(64u, 64u);
  constexpr RangeAggregatePlan metal_range = stencil::PlanStencilSourceVariant(
      RangeAggregateSourceVariant::Metal, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, requested,
      stencil::SourcePlanPath::SharedHalo);
  static_assert(metal_range.ok());
  const std::string metal = rund::node::accel::detail::MetalStencilSource(
      rund::kernel::StencilOp::Min,
      StencilGpuShapeFromRangeAggregatePlan(metal_range), metal_range);
  if (metal.find("rund_compute_stencil_min_i32") == std::string::npos ||
      metal.find("device const int* input") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  constexpr RangeAggregatePlan vulkan_range = stencil::PlanStencilSourceVariant(
      RangeAggregateSourceVariant::Vulkan, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, requested,
      stencil::SourcePlanPath::SharedHalo);
  static_assert(vulkan_range.ok());
  const std::string vulkan = rund::node::accel::detail::VulkanStencilSource(
      rund::kernel::StencilOp::Min, rund::kernel::StencilElement::U32,
      rund::kernel::ComputeDomain::I32,
      StencilGpuShapeFromRangeAggregatePlan(vulkan_range), vulkan_range);
  if (vulkan.find("int value = int(stencil_tile[center])") ==
      std::string::npos) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] constexpr bool StencilPhysicalDispatchHelpersAreExact() noexcept {
  using namespace rund::node::accel::detail;
  constexpr rund::kernel::u64 u32_groups =
      std::numeric_limits<rund::kernel::u32>::max();
  for (const rund::kernel::u32 width : kRangeAggregateWorkgroupWidths) {
    const StencilGpuShape shape = StencilGpuShape::shared(width, width);
    const rund::kernel::u64 u32_group_elements = u32_groups * width;
    const rund::kernel::u64 vulkan_u32_groups = StencilPhysicalGroupCount(
        std::numeric_limits<rund::kernel::u32>::max(), shape);
    if (!shape.valid() || shape.shared_element_capacity() != 3u * width ||
        StencilPhysicalGroupCount(0u, shape) != 0u ||
        StencilPhysicalGroupCount(1u, shape) != 1u ||
        StencilPhysicalGroupCount(width, shape) != 1u ||
        StencilPhysicalGroupCount(width + 1u, shape) != 2u ||
        !StencilPhysicalGroupsFit(7u * width, 7u, shape) ||
        StencilPhysicalGroupsFit(7u * width + 1u, 7u, shape) ||
        !StencilPhysicalGroupsFit(u32_group_elements, u32_groups, shape) ||
        StencilPhysicalGroupsFit(u32_group_elements + 1u, u32_groups, shape) ||
        StencilPhysicalGroupsFit(u32_group_elements + 1u, u32_groups + 1u,
                                 shape) ||
        !StencilVulkanDispatchFits(
            std::numeric_limits<rund::kernel::u32>::max(), vulkan_u32_groups,
            shape) ||
        StencilVulkanDispatchFits(
            static_cast<rund::kernel::u64>(
                std::numeric_limits<rund::kernel::u32>::max()) +
                1u,
            vulkan_u32_groups + 1u, shape) ||
        StencilPhysicalGroupsFit(1u, 0u, shape)) {
      return false;
    }
  }
  return true;
}

static_assert(StencilPhysicalDispatchHelpersAreExact());

[[nodiscard]] constexpr bool StencilFrozenPlanProjectionIsExact() noexcept {
  using namespace rund::node::accel::detail;
  constexpr std::uint8_t direct =
      RangeAggregateSupportBit(RangeAggregateSupport::Direct);
  constexpr std::uint8_t direct_shared =
      direct | RangeAggregateSupportBit(RangeAggregateSupport::SharedHalo);
  constexpr std::uint8_t direct_prefix =
      direct |
      RangeAggregateSupportBit(RangeAggregateSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      direct |
      RangeAggregateSupportBit(RangeAggregateSupport::BlockPrefixSuffix);
  constexpr auto shared_capabilities = RangeAggregateCapabilities::gpu(
      RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth128Bit, 128u, 4u,
      (128u + 2u * 7u) * 8u * 4u, std::numeric_limits<rund::kernel::u32>::max(),
      direct_shared);
  constexpr auto direct_capabilities = RangeAggregateCapabilities::gpu(
      RangeAggregateSourceVariant::Metal, kRangeAggregateWidth128Bit, 128u, 0u,
      0u, std::numeric_limits<rund::kernel::u32>::max(), direct);
  constexpr auto prefix_capabilities = RangeAggregateCapabilities::gpu(
      RangeAggregateSourceVariant::Metal, kRangeAggregateWidth64Bit, 64u, 4u,
      32768u, std::numeric_limits<rund::kernel::u32>::max(), direct_prefix);
  constexpr auto block_capabilities = RangeAggregateCapabilities::gpu(
      RangeAggregateSourceVariant::Vulkan, kRangeAggregateWidth64Bit, 64u, 0u,
      0u, std::numeric_limits<rund::kernel::u32>::max(), direct_block);
  constexpr auto sum_u64 =
      RangeAggregateTraits::sum_modulo(rund::kernel::ComputeDomain::U64);
  constexpr auto sum_u32 =
      RangeAggregateTraits::sum_modulo(rund::kernel::ComputeDomain::U32);
  constexpr auto minimum =
      RangeAggregateTraits::minimum(rund::kernel::ComputeDomain::I32);
  constexpr auto shared_shape = RangeAggregateShape::window(
      *sum_u64, RangeAggregateBoundary::Clamp, 129u, 7u, 8u);
  constexpr auto direct_shape = RangeAggregateShape::window(
      *sum_u32, RangeAggregateBoundary::Clamp, 65u, 65u, 4u);
  constexpr auto prefix_shape = RangeAggregateShape::window(
      *sum_u32, RangeAggregateBoundary::Clamp, 515u, 515u, 4u);
  constexpr auto block_shape = RangeAggregateShape::window(
      *minimum, RangeAggregateBoundary::Clamp, 515u, 515u, 4u);
  constexpr RangeAggregatePlan shared_plan =
      PlanRangeAggregate(*shared_shape, *shared_capabilities);
  constexpr RangeAggregatePlan direct_plan =
      PlanRangeAggregate(*direct_shape, *direct_capabilities);
  constexpr RangeAggregatePlan prefix_plan =
      PlanRangeAggregate(*prefix_shape, *prefix_capabilities);
  constexpr RangeAggregatePlan block_plan =
      PlanRangeAggregate(*block_shape, *block_capabilities);
  constexpr RangeAggregatePlan cpu_plan =
      PlanRangeAggregate(*direct_shape, RangeAggregateCapabilities::cpu());
  constexpr RangeAggregatePlan unavailable_plan = PlanRangeAggregate(
      *direct_shape, RangeAggregateCapabilities::unavailable());
  constexpr RangeAggregatePlan metal_direct_range =
      stencil::PlanStencilSourceVariant(
          RangeAggregateSourceVariant::Metal, rund::kernel::StencilOp::Sum,
          rund::kernel::ComputeDomain::U32, StencilGpuShape::direct(64u),
          stencil::SourcePlanPath::Direct);

  return shared_plan.ok() && direct_plan.ok() && prefix_plan.ok() &&
         block_plan.ok() && cpu_plan.ok() && !unavailable_plan.ok() &&
         StencilGpuShapeFromRangeAggregatePlan(shared_plan) ==
             StencilGpuShape::shared(128u, 7u) &&
         StencilGpuShapeFromRangeAggregatePlan(direct_plan) ==
             StencilGpuShape::direct(128u) &&
         StencilGpuShapeFromRangeAggregatePlan(prefix_plan) ==
             StencilGpuShape::direct(64u) &&
         StencilGpuShapeFromRangeAggregatePlan(block_plan) ==
             StencilGpuShape::direct(64u) &&
         !StencilGpuShapeFromRangeAggregatePlan(cpu_plan).valid() &&
         !StencilGpuShapeFromRangeAggregatePlan(unavailable_plan).valid() &&
         StencilElementBytes(rund::kernel::StencilElement::U32) == 4u &&
         StencilElementBytes(rund::kernel::StencilElement::U64) == 8u &&
         StencilElementBytes(static_cast<rund::kernel::StencilElement>(0u)) ==
             0u &&
         MetalStencilShapeCompiledPipelineSupport(
             StencilGpuShape::direct(64u), metal_direct_range,
             rund::kernel::StencilElement::U32,
             MetalStencilCompiledPipelineLimits{
                 .maximum_workgroup_width = 64u,
                 .static_shared_bytes = 4u,
                 .shared_memory_limit = 32768u,
             }) == MetalStencilCompiledPipelineSupport::Invalid;
}

static_assert(StencilFrozenPlanProjectionIsExact());

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
  const StencilBinds bindings{
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

[[nodiscard]] bool MetalStencilSourcePublicationIsTransactional() {
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

[[nodiscard]] bool MetalStencilRetainedSourceRetryIsExact() {
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
  // This is the CompileMetalStencilPipeline caller's exact failed-publication
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
  rund::node::accel::detail::StencilGpuShape shape{};
};

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
  const StencilBinds bindings{
      .input = &fixture.input.resident,
      .input_handle = &input_handle,
      .output = &fixture.output.resident,
      .output_handle = &output_handle,
  };
  StencilGpuShape maximum{};
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
      const RangeAggregatePlan range = MetalStencilRangePlan(
          admission.pick->raw, plan, rund::kernel::ComputeDomain::U32);
      std::shared_ptr<void> candidate;
      const rund::AccelCheck check = PrepareMetalStencil(
          admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
          bindings, range, candidate);
      const auto *const native =
          static_cast<const MetalStencilEncodeResources *>(candidate.get());
      if (!plan.ok || !range.ok() || !check.ok || native == nullptr ||
          !native->shape.valid() ||
          native->stage_count != range.stage_count() ||
          native->stage_count == 0u || native->pipelines[0u] == nullptr) {
        std::cerr << "metal maximum shared probe failed: radius=" << radius
                  << " check=" << check.ok << " reason=" << check.reason
                  << " native=" << (native != nullptr) << '\n';
        return {};
      }
      if (!native->shape.uses_shared_memory()) {
        continue;
      }
      if (native->shape.radius_cap() != radius ||
          input.size() % native->shape.width() != 3u ||
          StencilPhysicalGroupCount(input.size(), native->shape) <= 1u) {
        std::cerr << "metal maximum shared shape mismatch: radius=" << radius
                  << " width=" << native->shape.width()
                  << " cap=" << native->shape.radius_cap() << '\n';
        return {};
      }
      maximum = native->shape;
      selected_desc = desc;
      selected_plan = plan;
      first = std::move(candidate);
      break;
    }
    if (!maximum.valid()) {
      return {RuntimeSharedProbeStatus::NoSharedCapabilityVerified, {}};
    }

    const auto *const first_native =
        static_cast<const MetalStencilEncodeResources *>(first.get());
    MetalKernelImmutablePipelines immutable{};
    if (first_native->stage_count != 1u ||
        first_native->pipelines[0u] == nullptr) {
      return {};
    }
    immutable.stages[0u] = first_native->pipelines[0u];
    immutable.count = 1u;
    rund::node::accel::ResetRuntimeStats(fixture.context.pick);
    std::shared_ptr<void> second;
    const RangeAggregatePlan selected_range = MetalStencilRangePlan(
        admission.pick->raw, selected_plan, rund::kernel::ComputeDomain::U32);
    const rund::AccelCheck second_check =
        PrepareMetalStencil(admission.pick->raw, selected_desc, selected_plan,
                            rund::kernel::ComputeDomain::U32, bindings,
                            selected_range, second, &immutable);
    const auto *const second_native =
        static_cast<const MetalStencilEncodeResources *>(second.get());
    const rund::RuntimeStats stats =
        rund::node::accel::ReadRuntimeStats(fixture.context.pick);
    const bool matched =
        second_check.ok && second_native != nullptr &&
        second_native->shape.uses_shared_memory() &&
        second_native->shape == maximum && stats.ok &&
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
  if (!stencil::MatchesCapabilitySharedBoundaryU32(pick,
                                                   maximum.radius_cap())) {
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
  StencilGpuShape maximum{};
  for (rund::kernel::u64 radius = 256u; radius != 0u; --radius) {
    const rund::kernel::StencilPlan plan =
        rund::kernel::PlanStencil(rund::kernel::StencilDesc{
            .op = rund::kernel::StencilOp::Sum,
            .element = rund::kernel::StencilElement::U32,
            .boundary = rund::kernel::StencilBoundary::Clamp,
            .element_count = 515u,
            .radius = radius});
    const RangeAggregatePlan range = VulkanStencilRangePlan(
        token->raw, plan, rund::kernel::ComputeDomain::U32);
    const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(range);
    if (!shape.valid()) {
      std::cerr << "vulkan maximum shared probe invalid: radius=" << radius
                << '\n';
      return {};
    }
    if (!shape.uses_shared_memory()) {
      continue;
    }
    if (shape.radius_cap() != radius || 515u % shape.width() != 3u ||
        StencilPhysicalGroupCount(515u, shape) <= 1u) {
      std::cerr << "vulkan maximum shared shape mismatch: radius=" << radius
                << " width=" << shape.width() << " cap=" << shape.radius_cap()
                << '\n';
      return {};
    }
    maximum = shape;
    break;
  }
  if (!maximum.valid()) {
    return {RuntimeSharedProbeStatus::NoSharedCapabilityVerified, {}};
  }

  std::array<rund::kernel::u32, 515u> input{};
  auto fixture = stencil::match::BuildResources(
      pick, rund::kernel::ComputeScalar::Lane32,
      rund::kernel::ComputeDomain::U32, rund::kernel::StencilOp::Sum,
      rund::kernel::StencilElement::U32, maximum.radius_cap(), input);
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
      .radius = maximum.radius_cap(),
  };
  const rund::kernel::StencilPlan plan = rund::kernel::PlanStencil(desc);
  const StencilBinds bindings{
      .input = &fixture.input.resident,
      .input_handle = &input_handle,
      .output = &fixture.output.resident,
      .output_handle = &output_handle,
  };
  {
    const KernelPreparationScope preparation{
        KernelPreparationMode::PipelinePrivate};
    std::shared_ptr<void> first;
    const RangeAggregatePlan range = VulkanStencilRangePlan(
        admission.pick->raw, plan, rund::kernel::ComputeDomain::U32);
    const rund::AccelCheck first_check = PrepareVulkanStencil(
        admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
        bindings, range, first);
    const auto *const first_native =
        static_cast<const VulkanStencilEncodeResources *>(first.get());
    if (!plan.ok || !range.ok() || !first_check.ok || first_native == nullptr ||
        first_native->stage_count != range.stage_count() ||
        first_native->stage_count != 1u ||
        first_native->pipelines[0u] == nullptr ||
        !first_native->shape.uses_shared_memory() ||
        first_native->shape != maximum) {
      return {};
    }
    VulkanKernelImmutablePipelines immutable{};
    immutable.kind = rund::kernel::NodeKind::Stencil;
    if (!immutable.append(first_native->pipelines[0u],
                          StencilRangeDescriptorCount(range), 1u)) {
      return {};
    }
    rund::node::accel::ResetRuntimeStats(fixture.context.pick);
    std::shared_ptr<void> second;
    const rund::AccelCheck second_check = PrepareVulkanStencil(
        admission.pick->raw, desc, plan, rund::kernel::ComputeDomain::U32,
        bindings, range, second, &immutable);
    const auto *const second_native =
        static_cast<const VulkanStencilEncodeResources *>(second.get());
    const rund::RuntimeStats stats =
        rund::node::accel::ReadRuntimeStats(fixture.context.pick);
    if (!second_check.ok || second_native == nullptr ||
        !second_native->shape.uses_shared_memory() ||
        second_native->shape != maximum || second_native->stage_count != 1u ||
        second_native->pipelines[0u] != first_native->pipelines[0u] ||
        !stats.ok || stats.pipeline_compile_count != 0u ||
        stats.pipeline_cache_hit_count != 0u) {
      return {};
    }
  }
  if (!stencil::MatchesCapabilitySharedBoundaryU32(pick,
                                                   maximum.radius_cap())) {
    return {};
  }
  return {RuntimeSharedProbeStatus::SharedVerified, maximum};
}
#endif

[[nodiscard]] bool StencilSourcesCarryConvergentSharedHalo() {
  using namespace rund::node::accel::detail;
  for (const rund::kernel::u32 width : kRangeAggregateWorkgroupWidths) {
    const StencilGpuShape requested = StencilGpuShape::shared(width, width);
    const RangeAggregatePlan metal_range = stencil::PlanStencilSourceVariant(
        RangeAggregateSourceVariant::Metal, rund::kernel::StencilOp::Sum,
        rund::kernel::ComputeDomain::U64, requested,
        stencil::SourcePlanPath::SharedHalo);
    const StencilGpuShape shape =
        StencilGpuShapeFromRangeAggregatePlan(metal_range);
    if (!metal_range.ok() || shape != requested) {
      return false;
    }
    const std::string metal =
        MetalStencilSource(rund::kernel::StencilOp::Sum, shape, metal_range);
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
    const RangeAggregatePlan vulkan_range = stencil::PlanStencilSourceVariant(
        RangeAggregateSourceVariant::Vulkan, rund::kernel::StencilOp::Sum,
        rund::kernel::ComputeDomain::U64, requested,
        stencil::SourcePlanPath::SharedHalo);
    const StencilGpuShape vulkan_shape =
        StencilGpuShapeFromRangeAggregatePlan(vulkan_range);
    if (!vulkan_range.ok() || vulkan_shape != requested) {
      return false;
    }
    const std::string vulkan = VulkanStencilSource(
        rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U64,
        rund::kernel::ComputeDomain::U64, vulkan_shape, vulkan_range);
    const std::string vulkan_group =
        "layout(local_size_x = " + std::to_string(width) + ") in;";
    const std::string vulkan_tile =
        "shared uint64_t stencil_tile[" + std::to_string(3u * width) + "];";
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
    const std::string vulkan_u32 = VulkanStencilSource(
        rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U32,
        rund::kernel::ComputeDomain::U32, vulkan_shape,
        stencil::PlanStencilSourceVariant(
            RangeAggregateSourceVariant::Vulkan, rund::kernel::StencilOp::Sum,
            rund::kernel::ComputeDomain::U32, requested,
            stencil::SourcePlanPath::SharedHalo));
    const std::string vulkan_extended = vulkan + '\n';
    std::uint64_t vulkan_upper = 0u;
    if (!VulkanStencilSourceBytes(rund::kernel::StencilOp::Sum,
                                  rund::kernel::StencilElement::U64,
                                  rund::kernel::ComputeDomain::U64,
                                  vulkan_shape, vulkan_range, vulkan_upper)) {
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
        vulkan_u32.find("uint value = stencil_tile[center];") ==
            std::string::npos ||
        vulkan_u32.find("uint64_t value") != std::string::npos ||
        vulkan_u32.find("uint64_t(stencil_tile[center - step])") !=
            std::string::npos ||
        !VulkanStencilSourceMatches(
            rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U64,
            rund::kernel::ComputeDomain::U64, vulkan_shape, vulkan_range,
            vulkan, SourceHash(vulkan)) ||
        VulkanStencilSourceMatches(
            rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U64,
            rund::kernel::ComputeDomain::U64, vulkan_shape, vulkan_range,
            vulkan_extended, SourceHash(vulkan_extended)) ||
        VulkanStencilSourceMatches(
            rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U64,
            rund::kernel::ComputeDomain::U64, vulkan_shape, vulkan_range,
            vulkan, SourceHash(vulkan) ^ 1u) ||
        vulkan.size() != vulkan_upper || vulkan_u32.empty()) {
      return false;
    }
#endif
  }

  const StencilGpuShape direct_requested = StencilGpuShape::direct(64u);
  const RangeAggregatePlan metal_direct_range =
      stencil::PlanStencilSourceVariant(
          RangeAggregateSourceVariant::Metal, rund::kernel::StencilOp::Sum,
          rund::kernel::ComputeDomain::U32, direct_requested,
          stencil::SourcePlanPath::Direct);
  const StencilGpuShape direct =
      StencilGpuShapeFromRangeAggregatePlan(metal_direct_range);
  if (!metal_direct_range.ok() || direct != direct_requested) {
    return false;
  }
  const std::string metal_direct = MetalStencilSource(
      rund::kernel::StencilOp::Sum, direct, metal_direct_range);
  if (metal_direct.find("threadgroup uint tile[") != std::string::npos ||
      metal_direct.find("threadgroup_barrier") != std::string::npos ||
      metal_direct.find("for (ulong step = 1ul;") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const RangeAggregatePlan vulkan_direct_range =
      stencil::PlanStencilSourceVariant(
          RangeAggregateSourceVariant::Vulkan, rund::kernel::StencilOp::Sum,
          rund::kernel::ComputeDomain::U32, direct_requested,
          stencil::SourcePlanPath::Direct);
  if (!vulkan_direct_range.ok()) {
    return false;
  }
  const std::string vulkan_direct = VulkanStencilSource(
      rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U32,
      rund::kernel::ComputeDomain::U32,
      StencilGpuShapeFromRangeAggregatePlan(vulkan_direct_range),
      vulkan_direct_range);
  std::uint64_t direct_upper = 0u;
  if (!VulkanStencilSourceBytes(
          rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U32,
          rund::kernel::ComputeDomain::U32,
          StencilGpuShapeFromRangeAggregatePlan(vulkan_direct_range),
          vulkan_direct_range, direct_upper)) {
    return false;
  }
  if (vulkan_direct.find("shared uint stencil_tile[") != std::string::npos ||
      vulkan_direct.find("barrier();") != std::string::npos ||
      vulkan_direct.find("for (uint64_t step = uint64_t(1);") ==
          std::string::npos ||
      vulkan_direct.size() != direct_upper) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] bool RangeAggregateSourcesCarryLinearFamilies() {
  using namespace rund::node::accel::detail;
  constexpr StencilGpuShape physical = StencilGpuShape::direct(64u);
  constexpr RangeAggregatePlan metal_prefix = stencil::PlanStencilSourceVariant(
      RangeAggregateSourceVariant::Metal, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, physical,
      stencil::SourcePlanPath::PrefixDifference);
  constexpr RangeAggregatePlan metal_block = stencil::PlanStencilSourceVariant(
      RangeAggregateSourceVariant::Metal, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, physical,
      stencil::SourcePlanPath::BlockPrefixSuffix);
  constexpr RangeAggregatePlan metal_direct = stencil::PlanStencilSourceVariant(
      RangeAggregateSourceVariant::Metal, rund::kernel::StencilOp::Sum,
      rund::kernel::ComputeDomain::U32, physical,
      stencil::SourcePlanPath::Direct);
  static_assert(metal_prefix.ok() && metal_block.ok() && metal_direct.ok());
  static_assert(StencilRangeStaticSharedBytes(metal_prefix) == 64u * 4u);
  static_assert(StencilRangeStaticSharedBytes(metal_block) == 0u);
  if (metal_prefix.source_identity() == metal_direct.source_identity() ||
      metal_prefix.source_identity() == metal_block.source_identity() ||
      metal_block.source_identity() == metal_direct.source_identity()) {
    return false;
  }
  const std::string metal_prefix_source = MetalStencilSource(
      rund::kernel::StencilOp::Sum,
      StencilGpuShapeFromRangeAggregatePlan(metal_prefix), metal_prefix);
  const std::string metal_block_source = MetalStencilSource(
      rund::kernel::StencilOp::Min,
      StencilGpuShapeFromRangeAggregatePlan(metal_block), metal_block);
  std::uint64_t metal_prefix_upper = 0u;
  std::uint64_t metal_block_upper = 0u;
  if (!MetalStencilSourceUpperBytes(rund::kernel::StencilOp::Sum, metal_prefix,
                                    metal_prefix_upper) ||
      !MetalStencilSourceUpperBytes(rund::kernel::StencilOp::Min, metal_block,
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
      metal_block_source.find(
          "const ulong window = params.radius * 2ul + 1ul;") ==
          std::string::npos ||
      metal_block_source.find("backward_values[index]") == std::string::npos ||
      metal_block_source.find("output[i] = min(backward_values[i], "
                              "forward_values[right]);") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  constexpr RangeAggregatePlan vulkan_prefix =
      stencil::PlanStencilSourceVariant(
          RangeAggregateSourceVariant::Vulkan, rund::kernel::StencilOp::Sum,
          rund::kernel::ComputeDomain::U32, physical,
          stencil::SourcePlanPath::PrefixDifference);
  constexpr RangeAggregatePlan vulkan_block = stencil::PlanStencilSourceVariant(
      RangeAggregateSourceVariant::Vulkan, rund::kernel::StencilOp::Min,
      rund::kernel::ComputeDomain::I32, physical,
      stencil::SourcePlanPath::BlockPrefixSuffix);
  static_assert(vulkan_prefix.ok() && vulkan_block.ok());
  const StencilGpuShape vulkan_prefix_shape =
      StencilGpuShapeFromRangeAggregatePlan(vulkan_prefix);
  const StencilGpuShape vulkan_block_shape =
      StencilGpuShapeFromRangeAggregatePlan(vulkan_block);
  const std::string vulkan_prefix_source = VulkanStencilSource(
      rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U32,
      rund::kernel::ComputeDomain::U32, vulkan_prefix_shape, vulkan_prefix);
  const std::string vulkan_block_source = VulkanStencilSource(
      rund::kernel::StencilOp::Min, rund::kernel::StencilElement::U32,
      rund::kernel::ComputeDomain::I32, vulkan_block_shape, vulkan_block);
  std::uint64_t vulkan_prefix_upper = 0u;
  std::uint64_t vulkan_block_upper = 0u;
  if (!VulkanStencilSourceBytes(
          rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U32,
          rund::kernel::ComputeDomain::U32, vulkan_prefix_shape, vulkan_prefix,
          vulkan_prefix_upper) ||
      !VulkanStencilSourceBytes(
          rund::kernel::StencilOp::Min, rund::kernel::StencilElement::U32,
          rund::kernel::ComputeDomain::I32, vulkan_block_shape, vulkan_block,
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
      vulkan_block_source.find(
          "const uint64_t window = params.radius * uint64_t(2) + "
          "uint64_t(1);") == std::string::npos ||
      vulkan_block_source.find("scratch1_values[uint(block)]") ==
          std::string::npos ||
      !VulkanStencilSourceMatches(
          rund::kernel::StencilOp::Sum, rund::kernel::StencilElement::U32,
          rund::kernel::ComputeDomain::U32, vulkan_prefix_shape, vulkan_prefix,
          vulkan_prefix_source, SourceHash(vulkan_prefix_source)) ||
      !VulkanStencilSourceMatches(
          rund::kernel::StencilOp::Min, rund::kernel::StencilElement::U32,
          rund::kernel::ComputeDomain::I32, vulkan_block_shape, vulkan_block,
          vulkan_block_source, SourceHash(vulkan_block_source))) {
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
    if (probe.shape.valid() && probe.shape.uses_shared_memory() &&
        probe.shape.radius_cap() != 0u) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::NoSharedCapabilityVerified:
    if (!probe.shape.valid()) {
      return true;
    }
    break;
  case RuntimeSharedProbeStatus::Failed:
    break;
  }
  std::cerr << backend << " runtime maximum shared contract failed: status="
            << static_cast<unsigned>(probe.status)
            << " width=" << probe.shape.width()
            << " cap=" << probe.shape.radius_cap() << '\n';
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

[[nodiscard]] bool
BackendRunsForcedRangeAggregateParity(const rund::AccelDevice &pick) {
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
    return name.find("stencil.sum.u32.w") != std::string::npos;
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
    const RangeAggregatePlan range =
        token != nullptr
            ? VulkanStencilRangePlan(token->raw, plan, domain)
            : RangeAggregatePlan::rejected("compute_adapter_unavailable");
    const StencilGpuShape shape = StencilGpuShapeFromRangeAggregatePlan(range);
    if (!shape.valid()) {
      std::cerr << "vulkan stencil variant unavailable: " << name
                << " range=" << range.reason() << '\n';
      return false;
    }
    const RangeAggregateIdentity identity = range.source_identity();
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
                << name << " width=" << shape.width()
                << " radius_cap=" << shape.radius_cap() << " stages=" << stages
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
  std::array<rund::node::accel::detail::RangeAggregateIdentity, 8u> seen_{};
  std::size_t seen_count_{};
};
#endif

} // namespace

bool BackendRunsStencil(const rund::AccelDevice &pick) {
  return StencilMatch(StencilPhysicalDispatchHelpersAreExact(),
                      "shape.dispatch") &&
         StencilMatch(StencilFrozenPlanProjectionIsExact(),
                      "shape.projection") &&
         StencilMatch(MetalStencilRejectedCompileTelemetryIsExact(),
                      "metal.fallback-telemetry") &&
         StencilMatch(MetalStencilNamedPipelinePublicationIsTransactional(),
                      "metal.pipeline-publication") &&
         StencilMatch(MetalStencilSourcePublicationIsTransactional(),
                      "metal.source-publication") &&
         StencilMatch(MetalStencilRetainedSourceRetryIsExact(),
                      "metal.source-retry") &&
         StencilMatch(StencilShapeRejectsOnlyOverlappingStorage(),
                      "shape.storage") &&
         StencilMatch(StencilSourcesCarryConvergentSharedHalo(),
                      "source.variants") &&
         StencilMatch(RangeAggregateSourcesCarryLinearFamilies(),
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
      !StencilMatch(StencilPhysicalDispatchHelpersAreExact(),
                    "shape.dispatch") ||
      !StencilMatch(StencilFrozenPlanProjectionIsExact(), "shape.projection") ||
      !StencilMatch(MetalStencilRejectedCompileTelemetryIsExact(),
                    "metal.fallback-telemetry") ||
      !StencilMatch(MetalStencilNamedPipelinePublicationIsTransactional(),
                    "metal.pipeline-publication") ||
      !StencilMatch(MetalStencilSourcePublicationIsTransactional(),
                    "metal.source-publication") ||
      !StencilMatch(MetalStencilRetainedSourceRetryIsExact(),
                    "metal.source-retry") ||
      !StencilMatch(StencilShapeRejectsOnlyOverlappingStorage(),
                    "shape.storage") ||
      !StencilMatch(StencilSourcesCarryConvergentSharedHalo(),
                    "source.variants") ||
      !StencilMatch(RangeAggregateSourcesCarryLinearFamilies(),
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
  return BackendRunsStencilValueCases(pick) &&
         BackendRunsForcedRangeAggregateParity(pick) &&
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
      !StencilMatch(RangeAggregateSourcesCarryLinearFamilies(),
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
  return BackendRunsStencilValueCases(pick) &&
         BackendRunsForcedRangeAggregateParity(pick) &&
         StencilMatch(
             RuntimeSharedProbeMatchesContract(
                 VulkanMaximumSharedShapeContract(pick, *adapter), "vulkan"),
             "vulkan.maximum-shared-capability");
#else
  return false;
#endif
}

} // namespace node_accel_contract
