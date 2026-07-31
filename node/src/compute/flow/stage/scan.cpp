#include "../recipe.hpp"

#include "../../expression/state.hpp"
#include "../../type.hpp"

#include <array>
#include <limits>
#include <memory>
#include <utility>

namespace rund::compute::detail {

std::uint32_t flow_scan_value(const std::shared_ptr<FlowState> &flow,
                              const std::uint32_t input, const Scan scan) {
  if (flow == nullptr || !flow->status || input == 0u ||
      input > flow->values.size()) {
    return 0u;
  }
  const FlowValue &value = flow->values[input - 1u];
  const std::uint32_t output =
      append(*flow, value.type, value.count, value.fixed_format);
  if (output == 0u) {
    return 0u;
  }
  try {
    flow->steps.push_back(ScanStep{input, output, 0u, scan});
    return output;
  } catch (const std::bad_alloc &) {
    reject(*flow, Reason::FlowCapacity);
    return 0u;
  }
}

void flow_scan(const std::shared_ptr<FlowState> &flow, const Scan scan) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const std::uint32_t output = flow_scan_value(flow, flow->output, scan);
  if (output != 0u) {
    flow->output = output;
  }
}

std::uint32_t flow_bounded_scan_value(const std::shared_ptr<FlowState> &flow,
                                      const std::uint32_t input,
                                      const std::uint32_t count,
                                      const Scan scan) {
  if (flow == nullptr || !flow->status || input == 0u || count == 0u ||
      input > flow->values.size() || count > flow->values.size()) {
    return 0u;
  }
  const FlowValue &value = flow->values[input - 1u];
  const FlowValue &logical = flow->values[count - 1u];
  if (logical.count != 1u ||
      (logical.type != Type::U32 && logical.type != Type::U64)) {
    reject(*flow, Reason::BoundedCountInvalid);
    return 0u;
  }
  const std::uint32_t output =
      append(*flow, value.type, value.count, value.fixed_format);
  if (output == 0u) {
    return 0u;
  }
  flow->values[output - 1u].active = count;
  try {
    flow->steps.push_back(ScanStep{input, output, count, scan});
    return output;
  } catch (const std::bad_alloc &) {
    reject(*flow, Reason::FlowCapacity);
    return 0u;
  }
}

void flow_bounded_scan(const std::shared_ptr<FlowState> &flow,
                       const std::uint32_t count, const Scan scan) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const std::uint32_t output =
      flow_bounded_scan_value(flow, flow->output, count, scan);
  if (output != 0u) {
    flow->output = output;
  }
}

std::uint32_t flow_bounded_reduce_value(const std::shared_ptr<FlowState> &flow,
                                        const std::uint32_t input,
                                        const std::uint32_t count,
                                        const Reduce operation) {
  if (flow == nullptr || !flow->status || input == 0u || count == 0u ||
      input > flow->values.size() || count > flow->values.size()) {
    return 0u;
  }
  const FlowValue &value = flow->values[input - 1u];
  const FlowValue &logical = flow->values[count - 1u];
  if (logical.count != 1u ||
      (logical.type != Type::U32 && logical.type != Type::U64)) {
    reject(*flow, Reason::BoundedCountInvalid);
    return 0u;
  }
  const std::uint32_t output =
      append(*flow, value.type, 1u, value.fixed_format);
  if (output == 0u) {
    return 0u;
  }
  const std::array inputs{input, count};
  const std::array outputs{output};
  return append_primitive(*flow, inputs, outputs, Primitive::Reduce,
                          {.mode = static_cast<std::uint32_t>(operation)})
             ? output
             : 0u;
}

void flow_bounded_reduce(const std::shared_ptr<FlowState> &flow,
                         const std::uint32_t count, const Reduce operation) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const std::uint32_t output =
      flow_bounded_reduce_value(flow, flow->output, count, operation);
  if (output != 0u) {
    flow->output = output;
  }
}

std::uint32_t flow_bounded_sort_value(const std::shared_ptr<FlowState> &flow,
                                      const std::uint32_t input,
                                      const std::uint32_t count,
                                      const bool indices) {
  if (flow == nullptr || !flow->status || input == 0u || count == 0u ||
      input > flow->values.size() || count > flow->values.size()) {
    return 0u;
  }
  const FlowValue &value = flow->values[input - 1u];
  const FlowValue &logical = flow->values[count - 1u];
  if (logical.count != 1u ||
      (logical.type != Type::U32 && logical.type != Type::U64)) {
    reject(*flow, Reason::BoundedCountInvalid);
    return 0u;
  }
  const std::uint32_t output =
      append(*flow, indices ? Type::U32 : value.type, value.count,
             indices ? FixedFormat{} : value.fixed_format);
  if (output == 0u) {
    return 0u;
  }
  flow->values[output - 1u].active = count;
  const std::array inputs{input, count};
  const std::array outputs{output};
  return append_primitive(*flow, inputs, outputs,
                          indices ? Primitive::Argsort : Primitive::Sort, {})
             ? output
             : 0u;
}

void flow_bounded_sort(const std::shared_ptr<FlowState> &flow,
                       const std::uint32_t count, const bool indices) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const std::uint32_t output =
      flow_bounded_sort_value(flow, flow->output, count, indices);
  if (output != 0u) {
    flow->output = output;
  }
}

BoundedIds flow_filter_masks(const std::shared_ptr<FlowState> &flow,
                             const std::uint32_t input,
                             const std::uint32_t selected,
                             const std::uint32_t rejected) {
  if (flow == nullptr || !flow->status || input == 0u || selected == 0u ||
      rejected == 0u || input > flow->values.size() ||
      selected > flow->values.size() || rejected > flow->values.size()) {
    return {};
  }
  const FlowValue value = flow->values[input - 1u];
  const FlowValue selected_value = flow->values[selected - 1u];
  const FlowValue rejected_value = flow->values[rejected - 1u];
  if (selected_value.type != rejected_value.type ||
      selected_value.count != value.count ||
      rejected_value.count != value.count) {
    reject(*flow, Reason::GraphTypeMismatch);
    return {};
  }
  const Type count_type =
      type_bytes(selected_value.type) == sizeof(std::uint64_t) ? Type::U64
                                                               : Type::U32;
  const std::uint32_t count = append(*flow, count_type, 1u);
  const std::uint32_t values =
      append(*flow, value.type, value.count, value.fixed_format);
  if (count == 0u || values == 0u) {
    return {};
  }
  const std::array count_inputs{selected};
  const std::array count_outputs{count};
  if (!append_primitive(*flow, count_inputs, count_outputs, Primitive::Reduce,
                        {.flag = true})) {
    return {};
  }
  const std::array partition_inputs{rejected, input};
  const std::array partition_outputs{values};
  if (!append_primitive(*flow, partition_inputs, partition_outputs,
                        Primitive::Partition, {})) {
    return {};
  }
  flow->values[values - 1u].active = count;
  flow->values[count - 1u].parent = flow->values[input - 1u].active;
  return BoundedIds{values, count};
}

BoundedIds flow_filter_value(const std::shared_ptr<FlowState> &flow,
                             const std::uint32_t input, ExprRef selected,
                             ExprRef rejected) {
  if (flow == nullptr || !flow->status || selected.state == nullptr ||
      rejected.state == nullptr || !selected.state->status ||
      !rejected.state->status || input == 0u || input > flow->values.size()) {
    return {};
  }
  if (selected.type != rejected.type) {
    reject(*flow, Reason::GraphTypeMismatch);
    return {};
  }
  const std::array inputs{input};
  const std::array masks{selected, rejected};
  const ValueIds outputs = flow_map_multi(flow, inputs, "filter-flags", masks);
  return outputs.size() == masks.size()
             ? flow_filter_masks(flow, input, outputs[0u], outputs[1u])
             : BoundedIds{};
}

BoundedIds flow_filter(const std::shared_ptr<FlowState> &flow, ExprRef selected,
                       ExprRef rejected) {
  if (flow == nullptr || !flow->status) {
    return {};
  }
  const BoundedIds result = flow_filter_value(
      flow, flow->output, std::move(selected), std::move(rejected));
  if (result.values != 0u) {
    flow->output = result.values;
  }
  return result;
}

} // namespace rund::compute::detail
