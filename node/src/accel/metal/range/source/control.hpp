#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "../../range/local.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool EmitMetalRangeControlSource(
    Sink &sink,
    const RangePlan &plan) noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!plan.ok() || plan.source_variant() != RangeSource::Metal ||
      !plan.shape().resident_counted() || plan.stage_count() == 0u ||
      plan.stage_count() > kRangeStageCap) {
    return false;
  }
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

inline ulong rund_range_groups(const ulong count, const uint width) {
  return count == 0ul ? 0ul : 1ul + (count - 1ul) / ulong(width);
}

inline ulong rund_range_prefix(const ulong count, const uint level,
                               const uint width) {
  ulong value = count;
  for (uint index = 0u; index < level && value != 0ul; ++index) {
    value = rund_range_groups(value, width);
  }
  return value;
}

kernel void rund_range_control(
    device const uint *count_words [[buffer(0)]],
    device RangeParams *params [[buffer(1)]],
    device uint *indirect [[buffer(2)]],
    device uint2 *status [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid >= )MSL";
  (void)source.decimal(plan.stage_count());
  source += R"MSL(u) { return; }
  const ulong count = )MSL";
  if (plan.shape().count() == RangeCount::U64) {
    source += "ulong(count_words[0]) | (ulong(count_words[1]) << 32u)";
  } else {
    source += "ulong(count_words[0])";
  }
  source += R"MSL(;
  const bool valid = count <= )MSL";
  (void)source.decimal(plan.shape().input_count());
  source += R"MSL(ul;
  if (tid == 0u) { status[0] = uint2(valid ? 0u : 1u, 0u); }

  ulong elements = 0ul;
  ulong groups = 0ul;
  ulong auxiliary = 0ul;
  uint stage = 0u;
  uint width = 1u;
  if (valid && count != 0ul) {
)MSL";
  for (std::size_t index = 0u; index < plan.stage_count(); ++index) {
    const RangeStagePlan frozen = plan.stage(index);
    source += index == 0u ? "    if (tid == " : "    else if (tid == ";
    (void)source.decimal(index);
    source += R"MSL(u) {
      stage = )MSL";
    (void)source.decimal(static_cast<std::uint32_t>(frozen.disposition));
    source += R"MSL(u;
      width = )MSL";
    (void)source.decimal(frozen.width);
    source += R"MSL(u;
)MSL";
    switch (frozen.disposition) {
    case RangeStageKind::Direct:
    case RangeStageKind::SharedHalo:
    case RangeStageKind::PrefixWindow:
    case RangeStageKind::BlockWindow:
      source += R"MSL(      elements = count;
      groups = rund_range_groups(elements, width);
      auxiliary = groups;
)MSL";
      break;
    case RangeStageKind::PrefixBlock:
    case RangeStageKind::PrefixSummary:
      source += "      elements = rund_range_prefix(count, ";
      (void)source.decimal(frozen.level);
      source += R"MSL(u, width);
      groups = rund_range_groups(elements, width);
)MSL";
      if (frozen.level != 0u) {
        source += "      if (rund_range_prefix(count, ";
        (void)source.decimal(static_cast<std::uint32_t>(frozen.level - 1u));
        source += R"MSL(u, width) <= ulong(width)) { groups = 0ul; }
)MSL";
      }
      source += "      auxiliary = groups;\n";
      break;
    case RangeStageKind::PrefixFixup:
      source += "      elements = rund_range_prefix(count, ";
      (void)source.decimal(frozen.level);
      source += R"MSL(u, width);
      groups = rund_range_groups(elements, width);
      if (groups <= 1ul) { groups = 0ul; }
      auxiliary = groups;
)MSL";
      break;
    case RangeStageKind::BlockPrefixSuffix:
      source += "      elements = count - 1ul + ";
      (void)source.decimal(plan.shape().window_size());
      source += R"MSL(ul;
      auxiliary = rund_range_groups(elements, )MSL";
      (void)source.decimal(plan.shape().window_size());
      source += R"MSL(u);
      groups = rund_range_groups(auxiliary, width);
)MSL";
      break;
    case RangeStageKind::PrefixSequential:
      return false;
    }
    source += "    }\n";
  }
  source += R"MSL(  }

  RangeParams row;
  if (!valid || count == 0ul) {
    row.input_count = 0ul;
    row.output_count = 0ul;
    row.window_size = 0ul;
    row.stride = 0ul;
    row.padding = 0ul;
    row.stage_element_count = 0ul;
    row.stage_aux_count = 0ul;
    row.stage = 0u;
    row.reserved = 0u;
  } else {
    row.input_count = count;
    row.output_count = count;
    row.window_size = )MSL";
  (void)source.decimal(plan.shape().window_size());
  source += R"MSL(ul;
    row.stride = )MSL";
  (void)source.decimal(plan.shape().stride());
  source += R"MSL(ul;
    row.padding = )MSL";
  (void)source.decimal(plan.shape().padding());
  source += R"MSL(ul;
    row.stage_element_count = elements;
    row.stage_aux_count = auxiliary;
    row.stage = stage;
    row.reserved = 0u;
  }
  params[tid] = row;

  const uint at = tid * 8u;
  const ulong work = groups * ulong(width);
  indirect[at + 0u] = valid ? uint(groups) : 0u;
  indirect[at + 1u] = valid && groups != 0ul ? 1u : 0u;
  indirect[at + 2u] = valid && groups != 0ul ? 1u : 0u;
  indirect[at + 3u] = uint(work);
  indirect[at + 4u] = uint(work >> 32u);
  indirect[at + 5u] = 0u;
  indirect[at + 6u] = 0u;
  indirect[at + 7u] = 0u;
}
)MSL";
  return source.valid();
}

} // namespace rund::node::accel::detail
