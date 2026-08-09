#include "../../domain.hpp"
#include "local.hpp"

#include "../../kernel/backend/source_recipe.hpp"
#include "../../source/hash.hpp"
#include "source/algebra.hpp"
#include "source/block.hpp"
#include "source/control.hpp"
#include "source/direct.hpp"
#include "source/prefix.hpp"
#include "source/shared.hpp"

namespace rund::node::accel::detail {

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
namespace {

class MatchSink final {
public:
  explicit MatchSink(const std::string_view source) noexcept
      : remaining_(source) {}

  [[nodiscard]] bool append(const std::string_view value) noexcept {
    if (value.size() > remaining_.size() ||
        remaining_.substr(0u, value.size()) != value) {
      return false;
    }
    remaining_.remove_prefix(value.size());
    return true;
  }

  [[nodiscard]] bool complete() const noexcept { return remaining_.empty(); }

private:
  std::string_view remaining_{};
};

template <typename Sink>
[[nodiscard]] bool
EmitVulkanRangeSource(Sink &sink, const RangeExec &execution) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  const RangeOp op = execution.operation();
  const RangePath candidate = execution.candidate();
  const bool range_scratch = candidate == RangePath::PrefixDifference ||
                             candidate == RangePath::BlockPrefixSuffix;
  const bool wide = execution.wide_elements();
  const bool signed_values = execution.signed_values();
  const RangeExec &shape = execution;
  const char *const scalar = VulkanRangeScalar(wide, signed_values);
  if (!sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl() in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t input_count;
  uint64_t output_count;
  uint64_t window_size;
  uint64_t stride;
  uint64_t padding;
  uint64_t stage_element_count;
  uint64_t stage_aux_count;
  uint stage;
  uint reserved;
} params;
layout(set = 0, binding = 1, std430) readonly buffer Input {
)glsl") ||
      !sink.append("  ") || !sink.append(scalar) ||
      !sink.append(" input_values[];\n") || !sink.append(R"glsl(};
layout(set = 0, binding = 2, std430) buffer Output {
)glsl") ||
      !sink.append("  ") || !sink.append(scalar) ||
      !sink.append(" output_values[];\n") || !sink.append("};\n")) {
    return false;
  }
  if (range_scratch &&
      (!sink.append(
           R"glsl(layout(set = 0, binding = 3, std430) buffer Scratch0 {
  )glsl") ||
       !sink.append(scalar) || !sink.append(R"glsl( scratch0_values[];
};
layout(set = 0, binding = 4, std430) buffer Scratch1 {
  )glsl") ||
       !sink.append(scalar) || !sink.append(R"glsl( scratch1_values[];
};
)glsl"))) {
    return false;
  }
  if (candidate == RangePath::PrefixDifference &&
      (!sink.append("shared ") || !sink.append(scalar) ||
       !sink.append(" range_scan[") ||
       !backend_source_recipe::append_decimal(sink, shape.width()) ||
       !sink.append("];\n"))) {
    return false;
  }
  if (candidate == RangePath::BlockPrefixSuffix &&
      (!sink.append("#define value_type ") || !sink.append(scalar) ||
       !sink.append("\n"))) {
    return false;
  }
  if (shape.uses_shared_halo() &&
      (!sink.append("shared ") || !sink.append(scalar) ||
       !sink.append(" range_tile[") ||
       !backend_source_recipe::append_decimal(
           sink, shape.shared_element_capacity()) ||
       !sink.append("];\n"))) {
    return false;
  }
  if (execution.saturating_sum() && !EmitVulkanRangeSaturatingAlgebra(sink)) {
    return false;
  }
  if (candidate == RangePath::PrefixDifference) {
    return op == RangeOp::Sum &&
           EmitVulkanPrefixDifferenceBody(
               sink, wide, execution.plan().shape().boundary(), shape);
  }
  if (candidate == RangePath::BlockPrefixSuffix) {
    return op != RangeOp::Sum &&
           EmitVulkanBlockPrefixSuffixBody(sink, op,
                                           execution.plan().shape().boundary(),
                                           wide, signed_values, shape);
  }
  if (shape.uses_shared_halo()) {
    return EmitVulkanRangeSharedBody(sink, op, wide, signed_values,
                                     execution.saturating_sum(), shape);
  }
  return EmitVulkanRangeDirectBody(sink, op, wide, signed_values,
                                   execution.saturating_sum(),
                                   execution.plan().shape().boundary(), shape);
}

} // namespace

std::string VulkanRangeSource(const RangeExec &execution) {
  return backend_source_recipe::materialize(
      [&](auto &sink) { return EmitVulkanRangeSource(sink, execution); });
}

bool VulkanRangeSourceBytes(const RangeExec &execution,
                            std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanRangeSource(sink, execution);
      },
      bytes);
}

bool VulkanRangeSourceMatches(const RangeExec &execution,
                              const std::string_view source,
                              const std::uint64_t source_hash) noexcept {
  if (SourceHash(source) != source_hash) {
    return false;
  }
  MatchSink sink{source};
  return EmitVulkanRangeSource(sink, execution) && sink.complete();
}

std::string VulkanRangeControlSource(const RangePlan &plan) {
  return backend_source_recipe::materialize(
      [&](auto &sink) { return EmitVulkanRangeControlSource(sink, plan); });
}

bool VulkanRangeControlSourceBytes(const RangePlan &plan,
                                   std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanRangeControlSource(sink, plan);
      },
      bytes);
}

bool VulkanRangeControlSourceMatches(const RangePlan &plan,
                                     const std::string_view source,
                                     const std::uint64_t source_hash) noexcept {
  if (SourceHash(source) != source_hash) {
    return false;
  }
  MatchSink sink{source};
  return EmitVulkanRangeControlSource(sink, plan) && sink.complete();
}
#endif

} // namespace rund::node::accel::detail
