#include "../recipe.hpp"
#include "filter.hpp"

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

std::uint32_t flow_bounded_reduce_value(const std::shared_ptr<FlowState> &flow,
                                        const std::uint32_t input,
                                        const std::uint32_t count,
                                        const Reduce operation) {
  if (flow == nullptr || !flow->status || input == 0u || count == 0u ||
      input > flow->values.size() || count > flow->values.size()) {
    return 0u;
  }
  if (const auto folded = fold_filter_sum(flow, input, count, operation)) {
    return *folded;
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

} // namespace rund::compute::detail
