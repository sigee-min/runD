#include "../../domain.hpp"
#include "local.hpp"

#include "../../kernel/backend/source_recipe.hpp"
#include "../../source/hash.hpp"
#include "source/body.hpp"

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
[[nodiscard]] bool EmitVulkanStencilSource(
    Sink &sink, const rund::kernel::StencilOp op,
    const rund::kernel::StencilElement element,
    const rund::kernel::ComputeDomain domain, const StencilGpuShape shape,
    const RangeAggregatePlan
        &range) noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!shape.valid() || !range.ok()) {
    return false;
  }
  const RangeAggregateCandidateDisposition candidate =
      range.candidate().disposition();
  const bool range_scratch =
      candidate == RangeAggregateCandidateDisposition::PrefixDifference ||
      candidate == RangeAggregateCandidateDisposition::BlockPrefixSuffix;
  const bool wide = element == rund::kernel::StencilElement::U64;
  const bool signed_extrema =
      IsSignedDomain(domain) && op != rund::kernel::StencilOp::Sum;
  const char *const scalar =
      candidate == RangeAggregateCandidateDisposition::BlockPrefixSuffix &&
              signed_extrema
          ? (wide ? "int64_t" : "int")
          : (wide ? "uint64_t" : "uint");
  if (!sink.append(R"glsl(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = )glsl") ||
      !backend_source_recipe::append_decimal(sink, shape.width()) ||
      !sink.append(R"glsl() in;
layout(set = 0, binding = 0, std430) readonly buffer Params {
  uint64_t element_count;
  uint64_t radius;
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
  if (candidate == RangeAggregateCandidateDisposition::PrefixDifference &&
      (!sink.append("shared ") || !sink.append(scalar) ||
       !sink.append(" range_scan[") ||
       !backend_source_recipe::append_decimal(sink, shape.width()) ||
       !sink.append("];\n"))) {
    return false;
  }
  if (candidate == RangeAggregateCandidateDisposition::BlockPrefixSuffix &&
      (!sink.append("#define value_type ") || !sink.append(scalar) ||
       !sink.append("\n"))) {
    return false;
  }
  if (shape.uses_shared_memory() &&
      (!sink.append("shared ") || !sink.append(wide ? "uint64_t" : "uint") ||
       !sink.append(" stencil_tile[") ||
       !backend_source_recipe::append_decimal(
           sink, shape.shared_element_capacity()) ||
       !sink.append("];\n"))) {
    return false;
  }
  if (candidate == RangeAggregateCandidateDisposition::PrefixDifference) {
    return op == rund::kernel::StencilOp::Sum &&
           EmitVulkanPrefixDifferenceBody(sink, wide, shape);
  }
  if (candidate == RangeAggregateCandidateDisposition::BlockPrefixSuffix) {
    return op != rund::kernel::StencilOp::Sum &&
           EmitVulkanBlockPrefixSuffixBody(sink, op, shape);
  }
  return EmitVulkanStencilBody(sink, op, wide, signed_extrema, shape);
}

} // namespace

std::string VulkanStencilSource(const rund::kernel::StencilOp op,
                                const rund::kernel::StencilElement element,
                                const rund::kernel::ComputeDomain domain,
                                const StencilGpuShape shape,
                                const RangeAggregatePlan &range) {
  return backend_source_recipe::materialize([&](auto &sink) {
    return EmitVulkanStencilSource(sink, op, element, domain, shape, range);
  });
}

bool VulkanStencilSourceBytes(const rund::kernel::StencilOp op,
                              const rund::kernel::StencilElement element,
                              const rund::kernel::ComputeDomain domain,
                              const StencilGpuShape shape,
                              const RangeAggregatePlan &range,
                              std::uint64_t &bytes) noexcept {
  return backend_source_recipe::bytes(
      [&](backend_source_recipe::CountSink &sink) noexcept {
        return EmitVulkanStencilSource(sink, op, element, domain, shape, range);
      },
      bytes);
}

bool VulkanStencilSourceMatches(const rund::kernel::StencilOp op,
                                const rund::kernel::StencilElement element,
                                const rund::kernel::ComputeDomain domain,
                                const StencilGpuShape shape,
                                const RangeAggregatePlan &range,
                                const std::string_view source,
                                const std::uint64_t source_hash) noexcept {
  if (SourceHash(source) != source_hash) {
    return false;
  }
  MatchSink sink{source};
  return EmitVulkanStencilSource(sink, op, element, domain, shape, range) &&
         sink.complete();
}
#endif

} // namespace rund::node::accel::detail
