#include "../recipe.hpp"

#include <array>
#include <limits>
#include <memory>

namespace rund::compute::detail {

std::uint32_t flow_unary_value(const std::shared_ptr<FlowState> &flow,
                               const std::uint32_t input,
                               const Primitive operation,
                               const Type output_type, const std::size_t count,
                               const PrimitiveOptions options) {
  if (flow == nullptr || !flow->status || input == 0u ||
      input > flow->values.size()) {
    return 0u;
  }
  const FlowValue &source = flow->values[input - 1u];
  const FixedFormat fixed_format =
      is_fixed(output_type) && source.type == output_type ? source.fixed_format
                                                          : FixedFormat{};
  const std::uint32_t output = append(*flow, output_type, count, fixed_format);
  if (output == 0u) {
    return 0u;
  }
  const std::array inputs{input};
  const std::array outputs{output};
  return append_primitive(*flow, inputs, outputs, operation, options) ? output
                                                                      : 0u;
}

BoundedIds flow_compact_value(const std::shared_ptr<FlowState> &flow,
                              const std::uint32_t input,
                              const std::size_t capacity) {
  if (flow == nullptr || !flow->status || input == 0u ||
      input > flow->values.size()) {
    return {};
  }
  const FlowValue &source = flow->values[input - 1u];
  if (source.type != Type::U32 ||
      source.count > std::numeric_limits<std::uint32_t>::max()) {
    reject(*flow, Reason::CompactTempOverflow);
    return {};
  }
  const std::uint32_t parent = source.active;
  const std::uint32_t values =
      flow_unary_value(flow, input, Primitive::Compact, Type::U32, capacity,
                       {.first = capacity});
  const std::uint32_t count = flow_unary_value(flow, input, Primitive::Reduce,
                                               Type::U32, 1u, {.flag = true});
  if (values == 0u || count == 0u) {
    return {};
  }
  flow->values[count - 1u].guard = values;
  flow->values[values - 1u].active = count;
  flow->values[count - 1u].parent = parent;
  return BoundedIds{values, count};
}

void flow_unary(const std::shared_ptr<FlowState> &flow,
                const Primitive operation, const Type output_type,
                const std::size_t count, const PrimitiveOptions options) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const std::uint32_t output = flow_unary_value(flow, flow->output, operation,
                                                output_type, count, options);
  if (output != 0u) {
    flow->output = output;
  }
}

} // namespace rund::compute::detail
