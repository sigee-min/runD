#include "local.hpp"

#include "src/accel/context/internal/execution.hpp"
#include "src/accel/metal/range/local.hpp"
#include "src/accel/metal/range/pipeline/name.hpp"
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

[[nodiscard]] constexpr bool IdentityContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const RangeCaps metal =
      Gpu(RangeSource::Metal, kRangeWidth256Bit, 256u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangeCaps vulkan =
      Gpu(RangeSource::Vulkan, kRangeWidth256Bit, 256u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangePlan first =
      ContractPlanRange(Shape(RangeOp::Sum, 515u, 300u), metal);
  const RangePlan second =
      ContractPlanRange(Shape(RangeOp::Sum, 515u, 515u), metal);
  const RangePlan other_backend =
      ContractPlanRange(Shape(RangeOp::Sum, 515u, 300u), vulkan);
  return first.ok() && second.ok() && other_backend.ok() &&
         first.candidate() == second.candidate() &&
         first.source_identity() == second.source_identity() &&
         first.execution_identity() != second.execution_identity() &&
         first.source_identity() != other_backend.source_identity();
}

[[nodiscard]] constexpr bool AffineIdentityContract() {
  constexpr std::uint8_t direct_prefix =
      RangeSupportBit(RangeSupport::Direct) |
      RangeSupportBit(RangeSupport::PrefixDifference);
  const RangeCaps caps =
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix);
  const RangePlan first = ContractPlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 257u, 65u, 129u, 2u, 64u),
      caps);
  const RangePlan second =
      ContractPlanRange(AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 513u,
                                    129u, 257u, 2u, 128u),
                        caps);
  const RangePlan clipped = ContractPlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u),
      caps);
  return first.ok() && second.ok() && clipped.ok() &&
         first.candidate() == second.candidate() &&
         first.source_identity() == second.source_identity() &&
         first.execution_identity() != second.execution_identity() &&
         first.source_identity() != clipped.source_identity();
}

[[nodiscard]] constexpr bool BlockStrideIdentityContract() {
  const auto caps = Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                        std::numeric_limits<u32>::max(),
                        RangeSupportBit(RangeSupport::Direct) |
                            RangeSupportBit(RangeSupport::BlockPrefixSuffix));
  const auto dense =
      ContractPlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 257u,
                                    65u, 129u, 1u, 64u),
                        caps);
  const auto strided =
      ContractPlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 257u,
                                    65u, 129u, 2u, 64u),
                        caps);
  const auto other_stride =
      ContractPlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 257u,
                                    65u, 129u, 3u, 64u),
                        caps);
  return dense.ok() && strided.ok() && other_stride.ok() &&
         dense.source_identity() != strided.source_identity() &&
         strided.source_identity() == other_stride.source_identity() &&
         strided.execution_identity() != other_stride.execution_identity();
}
[[nodiscard]] bool MetalBlockStrideKeyContract() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  const auto caps = Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                        std::numeric_limits<u32>::max(),
                        RangeSupportBit(RangeSupport::Direct) |
                            RangeSupportBit(RangeSupport::BlockPrefixSuffix));
  std::array<std::string, 3u> keys{}, sources{};
  for (u64 stride = 1u; stride <= 3u; ++stride) {
    const auto plan =
        ContractPlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip,
                                      257u, 65u, 129u, stride, 64u),
                          caps);
    if (!plan.ok() ||
        plan.candidate().disposition() != RangePath::BlockPrefixSuffix) {
      return false;
    }
    keys[stride - 1u] = RangePipelineKey(RequireExec(plan));
    sources[stride - 1u] = MetalRangeSource(RequireExec(plan));
  }
  return keys[0] != keys[1] && keys[1] == keys[2] && sources[0] != sources[1] &&
         sources[1] == sources[2];
#else
  return true;
#endif
}

static_assert(BlockStrideIdentityContract());

static_assert(IdentityContract());
static_assert(AffineIdentityContract());

} // namespace

[[nodiscard]] bool MetalRejectedCompileTelemetryIsExact() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  RecordMetalUncachedLibraryCompile(adapter, 17u);
  RecordMetalUncachedPipelineCompile(adapter, 23u);
  return adapter.stats.library_compile_count == 1u &&
         adapter.stats.runtime.run.time.shader_compile_ns == 17u &&
         adapter.stats.runtime.run.allocations.pipeline_compile_count == 1u &&
         adapter.stats.runtime.run.time.pipeline_create_ns == 23u &&
         adapter.source_libraries.empty() && adapter.named_pipelines.empty();
}

[[nodiscard]] bool MetalNamedPipelinePublicationIsTransactional() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  const std::shared_ptr<void> first = std::make_shared<int>(1);
  const std::shared_ptr<void> contender = std::make_shared<int>(2);
  const MetalNamedPipelinePublishResult inserted =
      PublishMetalNamedPipeline(adapter, "range.contract", first, 11u);
  if (inserted.status != MetalNamedPipelinePublishStatus::Inserted ||
      inserted.pipeline != first || adapter.named_pipelines.size() != 1u ||
      adapter.stats.runtime.run.allocations.pipeline_compile_count != 1u ||
      adapter.stats.runtime.run.time.pipeline_create_ns != 11u) {
    return false;
  }
  const MetalNamedPipelinePublishResult existing =
      PublishMetalNamedPipeline(adapter, "range.contract", contender, 17u);
  if (existing.status != MetalNamedPipelinePublishStatus::Existing ||
      existing.pipeline != first || adapter.named_pipelines.size() != 1u ||
      adapter.stats.runtime.run.allocations.pipeline_compile_count != 1u ||
      adapter.stats.runtime.run.time.pipeline_create_ns != 11u) {
    return false;
  }
  adapter.fault_named_pipeline_publish_once.store(true,
                                                  std::memory_order_relaxed);
  const MetalNamedPipelinePublishResult failed = PublishMetalNamedPipeline(
      adapter, "range.capacity", std::make_shared<int>(3), 23u);
  return failed.status == MetalNamedPipelinePublishStatus::Failed &&
         failed.pipeline == nullptr && adapter.named_pipelines.size() == 1u &&
         adapter.stats.runtime.run.allocations.pipeline_compile_count == 1u &&
         adapter.stats.runtime.run.time.pipeline_create_ns == 11u;
}

[[nodiscard]] bool MetalSourcePublicationIsTransactional() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  const std::shared_ptr<void> baseline_owner = std::make_shared<int>(0);
  const MetalSourceLibraryPublishResult baseline =
      PublishMetalSourceLibrary(adapter, "range.baseline", baseline_owner, 5u);
  if (baseline.status != MetalSourceLibraryPublishStatus::Inserted ||
      baseline.library != baseline_owner) {
    return false;
  }
  const std::shared_ptr<void> failed_owner = std::make_shared<int>(1);
  adapter.fault_source_library_publish_once.store(true,
                                                  std::memory_order_relaxed);
  const MetalSourceLibraryPublishResult failed =
      PublishMetalSourceLibrary(adapter, "range.source", failed_owner, 7u);
  if (failed.status != MetalSourceLibraryPublishStatus::Failed ||
      failed.library != nullptr || adapter.source_libraries.size() != 1u ||
      adapter.source_libraries.front().library != baseline_owner ||
      adapter.stats.library_compile_count != 2u ||
      adapter.stats.runtime.run.time.shader_compile_ns != 12u ||
      adapter.stats.library_cache_hit_count != 0u ||
      std::string_view{adapter.last_error} != "compute_pipeline_capacity") {
    return false;
  }

  const std::shared_ptr<void> inserted_owner = std::make_shared<int>(2);
  const MetalSourceLibraryPublishResult inserted =
      PublishMetalSourceLibrary(adapter, "range.source", inserted_owner, 11u);
  if (inserted.status != MetalSourceLibraryPublishStatus::Inserted ||
      inserted.library != inserted_owner ||
      adapter.source_libraries.size() != 2u ||
      adapter.stats.library_compile_count != 3u ||
      adapter.stats.runtime.run.time.shader_compile_ns != 23u ||
      adapter.stats.library_cache_hit_count != 0u) {
    return false;
  }

  const MetalSourceLibraryPublishResult existing = PublishMetalSourceLibrary(
      adapter, "range.source", std::make_shared<int>(3), 13u);
  return existing.status == MetalSourceLibraryPublishStatus::Existing &&
         existing.library == inserted_owner &&
         adapter.source_libraries.size() == 2u &&
         adapter.stats.library_compile_count == 4u &&
         adapter.stats.runtime.run.time.shader_compile_ns == 36u &&
         adapter.stats.library_cache_hit_count == 1u;
}

[[nodiscard]] bool MetalSourceRetryIsExact() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  const std::shared_ptr<void> library = std::make_shared<int>(1);
  const MetalSourceLibraryPublishResult source =
      PublishMetalSourceLibrary(adapter, "range.retry.source", library, 11u);
  if (source.status != MetalSourceLibraryPublishStatus::Inserted ||
      source.library != library) {
    return false;
  }

  adapter.fault_named_pipeline_publish_once.store(true,
                                                  std::memory_order_relaxed);
  const MetalNamedPipelinePublishResult failed = PublishMetalNamedPipeline(
      adapter, "range.retry.pipeline", std::make_shared<int>(2), 13u);
  if (failed.status != MetalNamedPipelinePublishStatus::Failed ||
      failed.pipeline != nullptr || !adapter.named_pipelines.empty() ||
      adapter.source_libraries.size() != 1u ||
      adapter.stats.runtime.run.allocations.pipeline_compile_count != 0u) {
    return false;
  }
  // This is the CompileMetalRange caller's exact failed-publication
  // disposition: the constructed PSO is transient but still counted once.
  RecordMetalUncachedPipelineCompile(adapter, 13u);

  if (LookupMetalSourceLibrary(adapter, "range.retry.source") != library) {
    return false;
  }
  const std::shared_ptr<void> retry_pipeline = std::make_shared<int>(3);
  const MetalNamedPipelinePublishResult retried = PublishMetalNamedPipeline(
      adapter, "range.retry.pipeline", retry_pipeline, 17u);
  return retried.status == MetalNamedPipelinePublishStatus::Inserted &&
         retried.pipeline == retry_pipeline &&
         adapter.source_libraries.size() == 1u &&
         adapter.named_pipelines.size() == 1u &&
         adapter.stats.library_compile_count == 1u &&
         adapter.stats.library_cache_hit_count == 1u &&
         adapter.stats.runtime.run.time.shader_compile_ns == 11u &&
         adapter.stats.runtime.run.allocations.pipeline_compile_count == 2u &&
         adapter.stats.runtime.run.time.pipeline_create_ns == 30u;
}

bool CacheContract() {
  return IdentityContract() && AffineIdentityContract() &&
         BlockStrideIdentityContract() && MetalBlockStrideKeyContract() &&
         MetalRejectedCompileTelemetryIsExact() &&
         MetalNamedPipelinePublicationIsTransactional() &&
         MetalSourcePublicationIsTransactional() && MetalSourceRetryIsExact();
}

} // namespace node_accel_contract::range
