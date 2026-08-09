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

[[nodiscard]] bool AffineSourceContract() {
  constexpr std::uint8_t direct = RangeSupportBit(RangeSupport::Direct);
  constexpr std::uint8_t direct_prefix =
      direct | RangeSupportBit(RangeSupport::PrefixDifference);
  constexpr std::uint8_t direct_block =
      direct | RangeSupportBit(RangeSupport::BlockPrefixSuffix);
  const RangePlan direct_clip = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 3u, 2u, 5u, 5u, 4u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct));
  const RangePlan direct_clamp = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clamp, 3u, 2u, 5u, 5u, 4u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
          std::numeric_limits<u32>::max(), direct));
  const RangePlan prefix_clip = PlanRange(
      AffineShape(RangeOp::Sum, RangeBoundary::Clip, 257u, 65u, 129u, 2u, 64u),
      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 4u, 32768u,
          std::numeric_limits<u32>::max(), direct_prefix));
  const RangePlan block_clip =
      PlanRange(AffineShape(RangeOp::Minimum, RangeBoundary::Clip, 101u, 7u,
                            50u, 3u, 10u, 4u, ComputeDomain::I32),
                Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block));
  const std::optional<RangeTraits> saturating =
      RangeTraits::sum_saturating(ComputeDomain::Fixed);
  const std::optional<RangeShape> saturating_shape =
      saturating.has_value()
          ? RangeShape::affine(*saturating, RangeBoundary::Clamp, 5u, 3u, 3u,
                               2u, 1u, 4u)
          : std::nullopt;
  const RangePlan saturating_direct =
      saturating_shape.has_value()
          ? PlanRange(*saturating_shape,
                      Gpu(RangeSource::Metal, kRangeWidth64Bit, 64u, 0u, 0u,
                          std::numeric_limits<u32>::max(), direct))
          : RangePlan::rejected("compute_range_aggregate_shape_invalid");
  const auto direct_exec = RangeExec::from(direct_clip);
  const auto direct_clamp_exec = RangeExec::from(direct_clamp);
  const auto prefix_exec = RangeExec::from(prefix_clip);
  const auto block_exec = RangeExec::from(block_clip);
  const auto saturating_exec = RangeExec::from(saturating_direct);
  if (!direct_exec.has_value() || !direct_clamp_exec.has_value() ||
      !prefix_exec.has_value() || !block_exec.has_value() ||
      !saturating_exec.has_value()) {
    return false;
  }
  const std::string direct_source = MetalRangeSource(*direct_exec);
  const std::string direct_clamp_source = MetalRangeSource(*direct_clamp_exec);
  const std::string prefix_source = MetalRangeSource(*prefix_exec);
  const std::string block_source = MetalRangeSource(*block_exec);
  const std::string saturating_source = MetalRangeSource(*saturating_exec);
  if (direct_source.find("ulong input_count;") == std::string::npos ||
      direct_source.find("ulong output_count;") == std::string::npos ||
      direct_source.find("ulong window_size;") == std::string::npos ||
      direct_source.find("ulong stride;") == std::string::npos ||
      direct_source.find("ulong padding;") == std::string::npos ||
      direct_source.find("if (input_index >= params.input_count)") ==
          std::string::npos ||
      direct_source.find("valid = false;") == std::string::npos ||
      direct_clamp_source.find("input_index = params.input_count - 1ul;") ==
          std::string::npos ||
      prefix_source.find("const ulong anchor = i * params.stride;") ==
          std::string::npos ||
      prefix_source.find("left_missing") != std::string::npos ||
      block_source.find("const ulong window = params.window_size;") ==
          std::string::npos ||
      block_source.find("const ulong left = i * params.stride;") ==
          std::string::npos ||
      block_source.find("2147483647") == std::string::npos ||
      saturating_source.find("device const int* input") == std::string::npos ||
      saturating_source.find("rund_range_add_sat(value, sample)") ==
          std::string::npos ||
      saturating_source.find("slot < params.window_size") ==
          std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  const RangePlan vulkan_direct = PlanRange(
      direct_clip.shape(), Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u,
                               0u, std::numeric_limits<u32>::max(), direct));
  const RangePlan vulkan_block =
      PlanRange(block_clip.shape(),
                Gpu(RangeSource::Vulkan, kRangeWidth64Bit, 64u, 0u, 0u,
                    std::numeric_limits<u32>::max(), direct_block));
  const auto vulkan_direct_exec = RangeExec::from(vulkan_direct);
  const auto vulkan_exec = RangeExec::from(vulkan_block);
  if (!vulkan_direct_exec.has_value() || !vulkan_exec.has_value()) {
    return false;
  }
  const std::string direct_vulkan_source =
      VulkanRangeSource(*vulkan_direct_exec);
  const std::string source = VulkanRangeSource(*vulkan_exec);
  std::uint64_t bytes = 0u;
  if (!VulkanRangeSourceBytes(*vulkan_exec, bytes) || bytes != source.size() ||
      source.find("uint64_t output_count;") == std::string::npos ||
      source.find("const uint64_t left = block * params.stride;") ==
          std::string::npos ||
      direct_vulkan_source.find("if (input_index >= params.input_count)") ==
          std::string::npos ||
      direct_vulkan_source.find("valid = false;") == std::string::npos ||
      source.find("2147483647") == std::string::npos ||
      !VulkanRangeSourceMatches(*vulkan_exec, source, SourceHash(source))) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] bool StableSourceByteContract() {
  struct SourceCase final {
    RangeOp operation;
    ComputeDomain domain;
    RangePath path;
    u32 shared_radius_capacity;
    std::uint64_t metal_bytes;
    std::uint64_t metal_hash;
    std::uint64_t vulkan_bytes;
    std::uint64_t vulkan_hash;
  };
  constexpr std::array cases{
      SourceCase{RangeOp::Sum, ComputeDomain::U32, RangePath::Direct, 0u, 5782u,
                 5241732093872250891ull, 1860u, 15946521728726338142ull},
      SourceCase{RangeOp::Minimum, ComputeDomain::I32, RangePath::SharedHalo,
                 64u, 11048u, 13934788009075350063ull, 3193u,
                 5710807108626550770ull},
      SourceCase{RangeOp::Sum, ComputeDomain::U32, RangePath::PrefixDifference,
                 0u, 11880u, 16150613125859792719ull, 3739u,
                 12968811649224947642ull},
      SourceCase{RangeOp::Maximum, ComputeDomain::I64,
                 RangePath::BlockPrefixSuffix, 0u, 7504u,
                 4882305884707311087ull, 2493u, 14217763674358663737ull},
  };
  for (const SourceCase &entry : cases) {
    const RangePlan metal_plan = range::PlanSourceVariant(
        RangeSource::Metal, entry.operation, entry.domain, 64u,
        entry.shared_radius_capacity, entry.path);
    if (!metal_plan.ok()) {
      return false;
    }
    const std::string metal = MetalRangeSource(range::RequireExec(metal_plan));
    if (metal.size() != entry.metal_bytes ||
        SourceHash(metal) != entry.metal_hash ||
        metal.find("stencil") != std::string::npos) {
      return false;
    }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
    const RangePlan vulkan_plan = range::PlanSourceVariant(
        RangeSource::Vulkan, entry.operation, entry.domain, 64u,
        entry.shared_radius_capacity, entry.path);
    if (!vulkan_plan.ok()) {
      return false;
    }
    const std::string vulkan =
        VulkanRangeSource(range::RequireExec(vulkan_plan));
    if (vulkan.size() != entry.vulkan_bytes ||
        SourceHash(vulkan) != entry.vulkan_hash ||
        vulkan.find("stencil") != std::string::npos) {
      return false;
    }
#endif
  }
  return true;
}

} // namespace

[[nodiscard]] bool SignedSourcesCarryDomainOrder() {
  using namespace rund::node::accel::detail;
  constexpr RangePlan metal_range = range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Minimum, rund::kernel::ComputeDomain::I32,
      64u, 64u, RangePath::SharedHalo);
  static_assert(metal_range.ok());
  const std::string metal = rund::node::accel::detail::MetalRangeSource(
      range::RequireExec(metal_range));
  if (metal.find("rund_range_min_i32") == std::string::npos ||
      metal.find("device const int* input") == std::string::npos) {
    return false;
  }
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  constexpr RangePlan vulkan_range = range::PlanSourceVariant(
      RangeSource::Vulkan, RangeOp::Minimum, rund::kernel::ComputeDomain::I32,
      64u, 64u, RangePath::SharedHalo);
  static_assert(vulkan_range.ok());
  const std::string vulkan = rund::node::accel::detail::VulkanRangeSource(
      range::RequireExec(vulkan_range));
  if (vulkan.find("int value = range_tile[first]") == std::string::npos) {
    return false;
  }
#endif
  return true;
}

[[nodiscard]] bool SourcesCarryLinearFamilies() {
  using namespace rund::node::accel::detail;
  constexpr RangePlan metal_prefix = range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 64u,
      0u, RangePath::PrefixDifference);
  constexpr RangePlan metal_block = range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Minimum, rund::kernel::ComputeDomain::I32,
      64u, 0u, RangePath::BlockPrefixSuffix);
  constexpr RangePlan metal_direct = range::PlanSourceVariant(
      RangeSource::Metal, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 64u,
      0u, RangePath::Direct);
  static_assert(metal_prefix.ok() && metal_block.ok() && metal_direct.ok());
  static_assert(range::RequireExec(metal_prefix).static_shared_bytes() ==
                64u * 4u);
  static_assert(range::RequireExec(metal_block).static_shared_bytes() == 0u);
  if (metal_prefix.source_identity() == metal_direct.source_identity() ||
      metal_prefix.source_identity() == metal_block.source_identity() ||
      metal_block.source_identity() == metal_direct.source_identity()) {
    return false;
  }
  const std::string metal_prefix_source =
      MetalRangeSource(range::RequireExec(metal_prefix));
  const std::string metal_block_source =
      MetalRangeSource(range::RequireExec(metal_block));
  std::uint64_t metal_prefix_upper = 0u;
  std::uint64_t metal_block_upper = 0u;
  if (!MetalRangeSourceUpperBytes(range::RequireExec(metal_prefix),
                                  metal_prefix_upper) ||
      !MetalRangeSourceUpperBytes(range::RequireExec(metal_block),
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
  constexpr RangePlan vulkan_prefix = range::PlanSourceVariant(
      RangeSource::Vulkan, RangeOp::Sum, rund::kernel::ComputeDomain::U32, 64u,
      0u, RangePath::PrefixDifference);
  constexpr RangePlan vulkan_block = range::PlanSourceVariant(
      RangeSource::Vulkan, RangeOp::Minimum, rund::kernel::ComputeDomain::I32,
      64u, 0u, RangePath::BlockPrefixSuffix);
  static_assert(vulkan_prefix.ok() && vulkan_block.ok());
  const std::string vulkan_prefix_source =
      VulkanRangeSource(range::RequireExec(vulkan_prefix));
  const std::string vulkan_block_source =
      VulkanRangeSource(range::RequireExec(vulkan_block));
  std::uint64_t vulkan_prefix_upper = 0u;
  std::uint64_t vulkan_block_upper = 0u;
  if (!VulkanRangeSourceBytes(range::RequireExec(vulkan_prefix),
                              vulkan_prefix_upper) ||
      !VulkanRangeSourceBytes(range::RequireExec(vulkan_block),
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
      !VulkanRangeSourceMatches(range::RequireExec(vulkan_prefix),
                                vulkan_prefix_source,
                                SourceHash(vulkan_prefix_source)) ||
      !VulkanRangeSourceMatches(range::RequireExec(vulkan_block),
                                vulkan_block_source,
                                SourceHash(vulkan_block_source))) {
    return false;
  }
#endif
  return true;
}

bool SourceContract() {
  return AffineSourceContract() && SignedSourcesCarryDomainOrder() &&
         SourcesCarryLinearFamilies() && StableSourceByteContract() &&
         SharedHaloSourceContract();
}

} // namespace node_accel_contract::range
