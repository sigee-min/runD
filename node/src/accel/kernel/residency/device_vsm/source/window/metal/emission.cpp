#include "internal.hpp"

#include "../../../../../../metal/range/source/algebra.hpp"

namespace rund::node::accel::detail::device_vsm_window_source::metal {

[[nodiscard]] const char *scalar_type(const RangeExec &execution) noexcept {
  if (execution.wide_elements()) {
    return execution.signed_values() ? "long" : "ulong";
  }
  return execution.signed_values() ? "int" : "uint";
}

void append_direct(std::string &source, const RangeExec &execution,
                   const char *const type,
                   const DeviceVsmWindowFusion &fusion) {
  source += R"MSL(
    for (uint rund_page_local = rund_local;
         rund_page_local < rund_page_elements;
         rund_page_local += rund_group_width) {
      const ulong i = rund_page_begin + ulong(rund_page_local);
      const ulong anchor = i * params.stride;
      )MSL";
  source += type;
  source += R"MSL( value = )MSL";
  source += type;
  source += R"MSL((0);
      bool seeded = false;
      for (ulong slot = 0ul; slot < params.window_size; ++slot) {
        ulong input_index = 0ul;
        bool valid = true;
        if (slot < params.padding) {
          const ulong delta = params.padding - slot;
          if (anchor < delta) {
)MSL";
  source += execution.plan().shape().boundary() == RangeBoundary::Clamp
                ? "            input_index = 0ul;\n"
                : "            valid = false;\n";
  source += R"MSL(          } else {
            input_index = anchor - delta;
)MSL";
  source += execution.plan().shape().boundary() == RangeBoundary::Clamp
                ? R"MSL(            if (input_index >= params.input_count) {
              input_index = params.input_count - 1ul;
            }
)MSL"
                : R"MSL(            if (input_index >= params.input_count) {
              valid = false;
            }
)MSL";
  source += R"MSL(          }
        } else {
          const ulong delta = slot - params.padding;
          if (anchor >= params.input_count ||
              delta >= params.input_count - anchor) {
)MSL";
  source += execution.plan().shape().boundary() == RangeBoundary::Clamp
                ? "            input_index = params.input_count - 1ul;\n"
                : "            valid = false;\n";
  source += R"MSL(          } else {
            input_index = anchor + delta;
          }
        }
        if (!valid) { continue; }
        const )MSL";
  source += type;
  source += " sample = ";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input[input_index]", "input_index");
  source += R"MSL(;
        if (!seeded) {
          value = sample;
          seeded = true;
        } else {
)MSL";
  source += MetalRangeUpdate(execution.operation(), execution.saturating_sum());
  source += R"MSL(        }
      }
      output[i] = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "i");
  source += R"MSL(;
    }
)MSL";
}

void append_shared(std::string &source, const RangeExec &execution,
                   const char *const type,
                   const DeviceVsmWindowFusion &fusion) {
  const std::string capacity =
      std::to_string(execution.shared_radius_capacity());
  source += "  threadgroup ";
  source += type;
  source += " range_tile[";
  source += std::to_string(execution.shared_element_capacity());
  source += R"MSL(];
    const uint rund_page_groups =
        (rund_page_elements + rund_group_width - 1u) / rund_group_width;
    for (uint rund_page_group = 0u; rund_page_group < rund_page_groups;
         ++rund_page_group) {
      const uint rund_page_group_begin = rund_page_group * rund_group_width;
      const uint active_lanes =
          min(rund_page_elements - rund_page_group_begin, rund_group_width);
      const ulong group_base =
          rund_page_begin + ulong(rund_page_group_begin);
      const ulong group_end = group_base + ulong(active_lanes);
      const uint left_inputs = uint(min(group_base, params.padding));
      const uint right_inputs =
          group_end >= params.input_count
              ? 0u
              : uint(min(params.input_count - group_end, params.padding));
      if (rund_local < active_lanes) {
        const )MSL";
  source += type;
  source += " center_value = ";
  append_map_chain_expression(source, fusion, MapSlot::Before,
                              "input[group_base + ulong(rund_local)]",
                              "group_base + ulong(rund_local)");
  source += ";\n";
  source +=
      "        range_tile[" + capacity + "u + rund_local] = center_value;\n";
  source += R"MSL(        if (left_inputs == 0u && rund_local == 0u) {
          for (uint slot = 0u; ulong(slot) < params.padding; ++slot) {
)MSL";
  source += "            range_tile[" + capacity +
            "u - uint(params.padding) + slot] = center_value;\n";
  source += R"MSL(          }
        }
        if (right_inputs == 0u && rund_local + 1u == active_lanes) {
          for (uint slot = 0u; ulong(slot) < params.padding; ++slot) {
)MSL";
  source += "            range_tile[" + capacity +
            "u + active_lanes + slot] = center_value;\n";
  source += R"MSL(          }
        }
      }
      if (rund_local < left_inputs) {
        const )MSL";
  source += type;
  source += " left_value = ";
  append_map_chain_expression(
      source, fusion, MapSlot::Before,
      "input[group_base - ulong(left_inputs) + ulong(rund_local)]",
      "group_base - ulong(left_inputs) + ulong(rund_local)");
  source += ";\n";
  source += "        range_tile[" + capacity +
            "u - left_inputs + rund_local] = left_value;\n";
  source +=
      R"MSL(        if (rund_local == 0u && ulong(left_inputs) < params.padding) {
          for (uint slot = 0u;
               ulong(slot) < params.padding - ulong(left_inputs); ++slot) {
)MSL";
  source += "            range_tile[" + capacity +
            "u - uint(params.padding) + slot] = left_value;\n";
  source += R"MSL(          }
        }
      }
      if (rund_local < right_inputs) {
        const )MSL";
  source += type;
  source += " right_value = ";
  append_map_chain_expression(
      source, fusion, MapSlot::Before,
      "input[group_end + ulong(rund_local)]",
      "group_end + ulong(rund_local)");
  source += ";\n";
  source += "        range_tile[" + capacity +
            "u + active_lanes + rund_local] = right_value;\n";
  source += R"MSL(        if (rund_local + 1u == right_inputs &&
            ulong(right_inputs) < params.padding) {
          for (uint slot = right_inputs; ulong(slot) < params.padding;
               ++slot) {
)MSL";
  source += "            range_tile[" + capacity +
            "u + active_lanes + slot] = right_value;\n";
  source += R"MSL(          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (rund_local < active_lanes) {
        const ulong i = group_base + ulong(rund_local);
        const uint center = )MSL";
  source += capacity;
  source += R"MSL(u + rund_local;
        const uint first = center - uint(params.padding);
        )MSL";
  source += type;
  source += R"MSL( value = range_tile[first];
        for (ulong slot = 1ul; slot < params.window_size; ++slot) {
          const )MSL";
  source += type;
  source += R"MSL( sample = range_tile[first + uint(slot)];
)MSL";
  source += MetalRangeUpdate(execution.operation(), execution.saturating_sum());
  source += R"MSL(        }
        output[i] = )MSL";
  append_map_chain_expression(source, fusion, MapSlot::After, "value", "i");
  source += R"MSL(;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
)MSL";
}

void append_counters(std::string &source) {
  source += R"MSL(    threadgroup_barrier(mem_flags::mem_device |
                                         mem_flags::mem_threadgroup);
    if (rund_local == 0u) {
      atomic_fetch_add_explicit(&rund_vsm_result[0], 1u,
                                memory_order_relaxed);
      atomic_fetch_add_explicit(&rund_vsm_result[1], 1u,
                                memory_order_relaxed);
      atomic_fetch_add_explicit(&rund_vsm_result[2], 1u,
                                memory_order_relaxed);
      atomic_fetch_add_explicit(&rund_vsm_result[3], 1u,
                                memory_order_relaxed);
      atomic_fetch_add_explicit(&rund_vsm_result[4], 1u,
                                memory_order_relaxed);
      atomic_fetch_add_explicit(&rund_vsm_result[5], 1u,
                                memory_order_relaxed);
      const uint rund_first_page =
          rund_page == 0u ? 0u : rund_page - 1u;
      const uint rund_last_page =
          min(rund_page + 1u, rund_vsm.page_count - 1u);
      const uint rund_footprint =
          3u * (rund_page + 1u) + 5u * rund_first_page +
          7u * rund_last_page + 11u * rund_page_elements;
      if (rund_page + 1u < rund_vsm.page_count) {
        atomic_fetch_add_explicit(&rund_vsm_result[8], 1u,
                                  memory_order_relaxed);
      }
      atomic_fetch_add_explicit(&rund_vsm_result[9], rund_footprint,
                                memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device |
                        mem_flags::mem_threadgroup);
  }
}
)MSL";
}

} // namespace rund::node::accel::detail::device_vsm_window_source::metal
