#include "internal.hpp"

#include "../../../../../../vulkan/range/source/algebra.hpp"

#include <string_view>

namespace rund::node::accel::detail::device_vsm_window_source::vulkan {
namespace {

class StringSink final {
public:
  explicit StringSink(std::string &value) noexcept : value_(value) {}

  [[nodiscard]] bool append(const std::string_view text) {
    value_.append(text);
    return true;
  }

private:
  std::string &value_;
};

} // namespace

void append_update(std::string &source, const RangeExec &execution) {
  StringSink sink{source};
  (void)EmitVulkanRangeUpdate(sink, execution.operation(),
                              execution.saturating_sum());
}

bool append_saturating_algebra(std::string &source) {
  StringSink sink{source};
  return EmitVulkanRangeSaturatingAlgebra(sink);
}

void append_direct(std::string &source, const RangeExec &execution,
                   const char *const scalar,
                   const DeviceVsmWindowFusion &fusion) {
  source += R"GLSL(    for (uint rund_page_local = rund_local;
         rund_page_local < rund_page_elements;
         rund_page_local += gl_WorkGroupSize.x) {
      const uint gid = rund_page_begin + rund_page_local;
      const uint64_t anchor = uint64_t(gid) * params.stride;
      )GLSL";
  source += scalar;
  source += R"GLSL( value = )GLSL";
  source += scalar;
  source += R"GLSL((0);
      bool seeded = false;
      for (uint64_t slot = uint64_t(0); slot < params.window_size; ++slot) {
        uint64_t input_index = uint64_t(0);
        bool valid = true;
        if (slot < params.padding) {
          const uint64_t delta = params.padding - slot;
          if (anchor < delta) {
)GLSL";
  source += execution.plan().shape().boundary() == RangeBoundary::Clamp
                ? "            input_index = uint64_t(0);\n"
                : "            valid = false;\n";
  source += R"GLSL(          } else {
            input_index = anchor - delta;
)GLSL";
  source += execution.plan().shape().boundary() == RangeBoundary::Clamp
                ? R"GLSL(            if (input_index >= params.input_count) {
              input_index = params.input_count - uint64_t(1);
            }
)GLSL"
                : R"GLSL(            if (input_index >= params.input_count) {
              valid = false;
            }
)GLSL";
  source += R"GLSL(          }
        } else {
          const uint64_t delta = slot - params.padding;
          if (anchor >= params.input_count ||
              delta >= params.input_count - anchor) {
)GLSL";
  source +=
      execution.plan().shape().boundary() == RangeBoundary::Clamp
          ? "            input_index = params.input_count - uint64_t(1);\n"
          : "            valid = false;\n";
  source += R"GLSL(          } else {
            input_index = anchor + delta;
          }
        }
        if (!valid) { continue; }
        const )GLSL";
  source += scalar;
  source += " item = ";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input_values[uint(input_index)]", "input_index");
  source += R"GLSL(;
        if (!seeded) {
          value = item;
          seeded = true;
        } else {
)GLSL";
  append_update(source, execution);
  source += R"GLSL(        }
      }
      output_values[gid] = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "gid");
  source += R"GLSL(;
    }
)GLSL";
}

void append_shared(std::string &source, const RangeExec &execution,
                   const char *const scalar,
                   const DeviceVsmWindowFusion &fusion) {
  const std::string capacity =
      std::to_string(execution.shared_radius_capacity());
  source += R"GLSL(    const uint rund_page_groups =
        (rund_page_elements + gl_WorkGroupSize.x - 1u) / gl_WorkGroupSize.x;
    for (uint rund_page_group = 0u; rund_page_group < rund_page_groups;
         ++rund_page_group) {
      const uint rund_page_group_begin =
          rund_page_group * gl_WorkGroupSize.x;
      const uint active_lanes =
          min(rund_page_elements - rund_page_group_begin, gl_WorkGroupSize.x);
      const uint64_t group_base =
          uint64_t(rund_page_begin + rund_page_group_begin);
      const uint64_t group_end = group_base + uint64_t(active_lanes);
      const uint left_inputs = uint(min(group_base, params.padding));
      const uint right_inputs =
          group_end >= params.input_count
              ? 0u
              : uint(min(params.input_count - group_end, params.padding));
      if (rund_local < active_lanes) {
        const )GLSL";
  source += scalar;
  source += " center_value = ";
  append_map_chain_expression(
      source, fusion, MapSlot::Before,
      "input_values[uint(group_base + uint64_t(rund_local))]",
      "group_base + uint64_t(rund_local)");
  source += ";\n";
  source +=
      "        range_tile[" + capacity + "u + rund_local] = center_value;\n";
  source += R"GLSL(        if (left_inputs == 0u && rund_local == 0u) {
          for (uint slot = 0u; uint64_t(slot) < params.padding; ++slot) {
)GLSL";
  source += "            range_tile[" + capacity +
            "u - uint(params.padding) + slot] = center_value;\n";
  source += R"GLSL(          }
        }
        if (right_inputs == 0u && rund_local + 1u == active_lanes) {
          for (uint slot = 0u; uint64_t(slot) < params.padding; ++slot) {
)GLSL";
  source += "            range_tile[" + capacity +
            "u + active_lanes + slot] = center_value;\n";
  source += R"GLSL(          }
        }
      }
      if (rund_local < left_inputs) {
        const )GLSL";
  source += scalar;
  source += " left_value = ";
  append_map_chain_expression(
      source, fusion, MapSlot::Before,
      "input_values[uint(group_base - uint64_t(left_inputs) + "
      "uint64_t(rund_local))]",
      "group_base - uint64_t(left_inputs) + uint64_t(rund_local)");
  source += ";\n";
  source += "        range_tile[" + capacity +
            "u - left_inputs + rund_local] = left_value;\n";
  source +=
      R"GLSL(        if (rund_local == 0u && uint64_t(left_inputs) < params.padding) {
          for (uint slot = 0u;
               uint64_t(slot) < params.padding - uint64_t(left_inputs);
               ++slot) {
)GLSL";
  source += "            range_tile[" + capacity +
            "u - uint(params.padding) + slot] = left_value;\n";
  source += R"GLSL(          }
        }
      }
      if (rund_local < right_inputs) {
        const )GLSL";
  source += scalar;
  source += " right_value = ";
  append_map_chain_expression(
      source, fusion, MapSlot::Before,
      "input_values[uint(group_end + uint64_t(rund_local))]",
      "group_end + uint64_t(rund_local)");
  source += ";\n";
  source += "        range_tile[" + capacity +
            "u + active_lanes + rund_local] = right_value;\n";
  source += R"GLSL(        if (rund_local + 1u == right_inputs &&
            uint64_t(right_inputs) < params.padding) {
          for (uint slot = right_inputs; uint64_t(slot) < params.padding;
               ++slot) {
)GLSL";
  source += "            range_tile[" + capacity +
            "u + active_lanes + slot] = right_value;\n";
  source += R"GLSL(          }
        }
      }
      barrier();
      if (rund_local < active_lanes) {
        const uint gid = uint(group_base + uint64_t(rund_local));
        const uint center = )GLSL";
  source += capacity;
  source += R"GLSL(u + rund_local;
        const uint first = center - uint(params.padding);
        )GLSL";
  source += scalar;
  source += R"GLSL( value = range_tile[first];
        for (uint64_t slot = uint64_t(1); slot < params.window_size; ++slot) {
          const )GLSL";
  source += scalar;
  source += R"GLSL( item = range_tile[first + uint(slot)];
)GLSL";
  append_update(source, execution);
  source += R"GLSL(        }
        output_values[gid] = )GLSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "gid");
  source += R"GLSL(;
      }
      barrier();
    }
)GLSL";
}

void append_counters(std::string &source) {
  source += R"GLSL(    memoryBarrierBuffer();
    barrier();
    if (rund_local == 0u) {
      atomicAdd(rund_vsm_result[0], 1u);
      atomicAdd(rund_vsm_result[1], 1u);
      atomicAdd(rund_vsm_result[2], 1u);
      atomicAdd(rund_vsm_result[3], 1u);
      atomicAdd(rund_vsm_result[4], 1u);
      atomicAdd(rund_vsm_result[5], 1u);
      const uint rund_first_page =
          rund_page == 0u ? 0u : rund_page - 1u;
      const uint rund_last_page =
          min(rund_page + 1u, rund_vsm.page_count - 1u);
      const uint rund_footprint =
          3u * (rund_page + 1u) + 5u * rund_first_page +
          7u * rund_last_page + 11u * rund_page_elements;
      if (rund_page + 1u < rund_vsm.page_count) {
        atomicAdd(rund_vsm_result[8], 1u);
      }
      atomicAdd(rund_vsm_result[9], rund_footprint);
    }
    memoryBarrierBuffer();
    barrier();
  }
}
)GLSL";
}

} // namespace rund::node::accel::detail::device_vsm_window_source::vulkan
