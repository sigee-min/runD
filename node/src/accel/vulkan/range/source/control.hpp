#pragma once

#include "../../../kernel/backend/source_recipe.hpp"
#include "../../range/local.hpp"

namespace rund::node::accel::detail {

template <typename Sink>
[[nodiscard]] bool EmitVulkanRangeControlSource(
    Sink &sink,
    const RangePlan &plan) noexcept(noexcept(sink.append(std::string_view{}))) {
  if (!plan.ok() || plan.source_variant() != RangeSource::Vulkan ||
      !plan.shape().resident_counted() || plan.stage_count() == 0u ||
      plan.stage_count() > kRangeStageCap) {
    return false;
  }
  backend_source_recipe::SourceBuilder<Sink> source{sink};
  source += R"GLSL(#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(local_size_x = 1) in;

struct RangeParams {
  uint64_t input_count;
  uint64_t output_count;
  uint64_t window_size;
  uint64_t stride;
  uint64_t padding;
  uint64_t stage_element_count;
  uint64_t stage_aux_count;
  uint stage;
  uint reserved;
};

layout(set = 0, binding = 0, std430) readonly buffer Count {
  uint count_words[];
};
layout(set = 0, binding = 1, std430) buffer Params {
  uint params_words[];
};
layout(set = 0, binding = 2, std430) buffer Indirect {
  uint indirect[];
};
layout(set = 0, binding = 3, std430) buffer Status {
  uvec2 status[];
};
layout(push_constant) uniform Push {
  uint count_word;
  uint param_stride_words;
} push;

void rund_range_store64(uint at, uint64_t value) {
  params_words[at] = uint(value);
  params_words[at + 1u] = uint(value >> 32u);
}

void rund_range_store_params(uint base, RangeParams row) {
  rund_range_store64(base + 0u, row.input_count);
  rund_range_store64(base + 2u, row.output_count);
  rund_range_store64(base + 4u, row.window_size);
  rund_range_store64(base + 6u, row.stride);
  rund_range_store64(base + 8u, row.padding);
  rund_range_store64(base + 10u, row.stage_element_count);
  rund_range_store64(base + 12u, row.stage_aux_count);
  params_words[base + 14u] = row.stage;
  params_words[base + 15u] = row.reserved;
}

uint64_t rund_range_groups(uint64_t count, uint width) {
  return count == uint64_t(0) ? uint64_t(0)
                              : uint64_t(1) +
                                    (count - uint64_t(1)) / uint64_t(width);
}

uint64_t rund_range_prefix(uint64_t count, uint level, uint width) {
  uint64_t value = count;
  for (uint index = 0u; index < level && value != uint64_t(0); ++index) {
    value = rund_range_groups(value, width);
  }
  return value;
}

void main() {
  const uint tid = gl_GlobalInvocationID.x;
  if (tid >= )GLSL";
  (void)source.decimal(plan.stage_count());
  source += R"GLSL(u) { return; }
  const uint64_t count = )GLSL";
  if (plan.shape().count() == RangeCount::U64) {
    source += R"GLSL(uint64_t(count_words[push.count_word]) |
      (uint64_t(count_words[push.count_word + 1u]) << 32u))GLSL";
  } else {
    source += "uint64_t(count_words[push.count_word])";
  }
  source += R"GLSL(;
  const bool valid = count <= )GLSL";
  (void)source.decimal(plan.shape().input_count());
  source += R"GLSL(ul;
  if (tid == 0u) { status[0] = uvec2(valid ? 0u : 1u, 0u); }

  uint64_t elements = uint64_t(0);
  uint64_t groups = uint64_t(0);
  uint64_t auxiliary = uint64_t(0);
  uint stage = 0u;
  uint width = 1u;
  if (valid && count != uint64_t(0)) {
)GLSL";
  for (std::size_t index = 0u; index < plan.stage_count(); ++index) {
    const RangeStagePlan frozen = plan.stage(index);
    source += index == 0u ? "    if (tid == " : "    else if (tid == ";
    (void)source.decimal(index);
    source += R"GLSL(u) {
      stage = )GLSL";
    (void)source.decimal(static_cast<std::uint32_t>(frozen.disposition));
    source += R"GLSL(u;
      width = )GLSL";
    (void)source.decimal(frozen.width);
    source += R"GLSL(u;
)GLSL";
    switch (frozen.disposition) {
    case RangeStageKind::Direct:
    case RangeStageKind::SharedHalo:
    case RangeStageKind::PrefixWindow:
    case RangeStageKind::BlockWindow:
      source += R"GLSL(      elements = count;
      groups = rund_range_groups(elements, width);
      auxiliary = groups;
)GLSL";
      break;
    case RangeStageKind::PrefixBlock:
    case RangeStageKind::PrefixSummary:
      source += "      elements = rund_range_prefix(count, ";
      (void)source.decimal(frozen.level);
      source += R"GLSL(u, width);
      groups = rund_range_groups(elements, width);
)GLSL";
      if (frozen.level != 0u) {
        source += "      if (rund_range_prefix(count, ";
        (void)source.decimal(static_cast<std::uint32_t>(frozen.level - 1u));
        source += R"GLSL(u, width) <= uint64_t(width)) {
        groups = uint64_t(0);
      }
)GLSL";
      }
      source += "      auxiliary = groups;\n";
      break;
    case RangeStageKind::PrefixFixup:
      source += "      elements = rund_range_prefix(count, ";
      (void)source.decimal(frozen.level);
      source += R"GLSL(u, width);
      groups = rund_range_groups(elements, width);
      if (groups <= uint64_t(1)) { groups = uint64_t(0); }
      auxiliary = groups;
)GLSL";
      break;
    case RangeStageKind::BlockPrefixSuffix:
      source += "      elements = count - uint64_t(1) + ";
      (void)source.decimal(plan.shape().window_size());
      source += R"GLSL(ul;
      auxiliary = rund_range_groups(elements, )GLSL";
      (void)source.decimal(plan.shape().window_size());
      source += R"GLSL(u);
      groups = rund_range_groups(auxiliary, width);
)GLSL";
      break;
    case RangeStageKind::PrefixSequential:
      return false;
    }
    source += "    }\n";
  }
  source += R"GLSL(  }

  RangeParams row;
  if (!valid || count == uint64_t(0)) {
    row.input_count = uint64_t(0);
    row.output_count = uint64_t(0);
    row.window_size = uint64_t(0);
    row.stride = uint64_t(0);
    row.padding = uint64_t(0);
    row.stage_element_count = uint64_t(0);
    row.stage_aux_count = uint64_t(0);
    row.stage = 0u;
    row.reserved = 0u;
  } else {
    row.input_count = count;
    row.output_count = count;
    row.window_size = )GLSL";
  (void)source.decimal(plan.shape().window_size());
  source += R"GLSL(ul;
    row.stride = )GLSL";
  (void)source.decimal(plan.shape().stride());
  source += R"GLSL(ul;
    row.padding = )GLSL";
  (void)source.decimal(plan.shape().padding());
  source += R"GLSL(ul;
    row.stage_element_count = elements;
    row.stage_aux_count = auxiliary;
    row.stage = stage;
    row.reserved = 0u;
  }
  rund_range_store_params(tid * push.param_stride_words, row);

  const uint at = tid * 8u;
  const uint64_t work = groups * uint64_t(width);
  indirect[at + 0u] = valid ? uint(groups) : 0u;
  indirect[at + 1u] = valid && groups != uint64_t(0) ? 1u : 0u;
  indirect[at + 2u] = valid && groups != uint64_t(0) ? 1u : 0u;
  indirect[at + 3u] = uint(work);
  indirect[at + 4u] = uint(work >> 32u);
  indirect[at + 5u] = 0u;
  indirect[at + 6u] = 0u;
  indirect[at + 7u] = 0u;
}
)GLSL";
  return source.valid();
}

} // namespace rund::node::accel::detail
