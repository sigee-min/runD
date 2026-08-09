#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "block.hpp"
#include "direct.hpp"
#include "prefix.hpp"
#include "shared.hpp"

#include <string_view>

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool
EmitMetalRangeSource(Sink &sink, const RangeExec &execution) noexcept(
    noexcept(sink.append(std::string_view{}))) {
  const RangeOp op = execution.operation();
  const RangeExec &shape = execution;
  const RangePath candidate = execution.candidate();
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  source += R"MSL(
#include <metal_stdlib>
using namespace metal;

struct RangeParams {
  ulong input_count;
  ulong output_count;
  ulong window_size;
  ulong stride;
  ulong padding;
  ulong stage_element_count;
  ulong stage_aux_count;
  uint stage;
  uint reserved;
};

)MSL";
  if (execution.saturating_sum()) {
    AppendMetalRangeSaturatingAlgebra(source);
  }
  if (candidate == RangePath::PrefixDifference) {
    if (op != RangeOp::Sum) {
      return false;
    }
    const RangeBoundary boundary = execution.plan().shape().boundary();
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "uint", "u32");
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "ulong", "u64");
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "uint", "i32");
    AppendMetalPrefixDifferenceKernel(source, boundary, shape, "ulong", "i64");
  } else if (candidate == RangePath::BlockPrefixSuffix) {
    if (op == RangeOp::Sum) {
      return false;
    }
    const RangeBoundary boundary = execution.plan().shape().boundary();
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "uint",
                                       "u32", MetalRangeIdentity(op, "u32"));
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "ulong",
                                       "u64", MetalRangeIdentity(op, "u64"));
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "int",
                                       "i32", MetalRangeIdentity(op, "i32"));
    AppendMetalBlockPrefixSuffixKernel(source, op, boundary, shape, "long",
                                       "i64", MetalRangeIdentity(op, "i64"));
  } else {
    const RangeBoundary boundary = execution.plan().shape().boundary();
    const bool saturating = execution.saturating_sum();
    const auto append = [&](const char *const type, const char *const suffix) {
      if (shape.uses_shared_halo()) {
        AppendMetalRangeSharedKernel(source, op, saturating, shape, type,
                                     suffix);
      } else {
        AppendMetalRangeDirectKernel(source, op, boundary, saturating, shape,
                                     type, suffix);
      }
    };
    append(saturating ? "int" : "uint", "u32");
    append(saturating ? "long" : "ulong", "u64");
    append(op == RangeOp::Sum && !saturating ? "uint" : "int", "i32");
    append(op == RangeOp::Sum && !saturating ? "ulong" : "long", "i64");
  }
  return source.valid();
}

} // namespace rund::node::accel::detail
