#include "recipe.hpp"

#include "../type.hpp"

#include <rund/compute/abi/expression.hpp>

#include <array>
#include <memory>
#include <string_view>
#include <utility>

namespace rund::compute::detail {

namespace {

std::uint32_t flow_retype_value(const std::shared_ptr<FlowState> &flow,
                                const std::uint32_t input,
                                const Type output_type,
                                const FixedFormat output_format,
                                const bool boundary_mask) {
  if (flow == nullptr || !flow->status || input == 0u ||
      input > flow->values.size()) {
    return 0u;
  }
  const FlowValue source = flow->values[input - 1u];
  const bool canonical_unsigned_mask =
      boundary_mask && !type_fixed(source.type) &&
      (output_type == Type::U32 || output_type == Type::U64) &&
      output_format == FixedFormat{};
  if (type_bytes(source.type) != type_bytes(output_type) &&
      !canonical_unsigned_mask) {
    reject(*flow, Reason::GraphTypeMismatch);
    return 0u;
  }
  const auto expression = make_expr();
  ExprRef value =
      detail::input(expression, source.type, 0u, source.fixed_format);
  if (canonical_unsigned_mask) {
    const ExprRef zero = constant(expression, source.type, 0u);
    value = make_mask(binary(ExprOp::NotEqual, std::move(value), zero),
                      output_type);
  } else {
    value = boundary_mask ? boundary_mask_expr(std::move(value), output_type,
                                               output_format)
                          : checked_ordinal_expr(std::move(value), output_type);
  }
  const std::uint32_t output =
      append(*flow, output_type, source.count, output_format);
  if (output == 0u) {
    return 0u;
  }
  const std::array inputs{input};
  const std::array outputs{output};
  const std::array expressions{value};
  return append_map(*flow, "retype", inputs, outputs, expressions) ? output
                                                                   : 0u;
}

} // namespace

std::uint32_t flow_retype(const std::shared_ptr<FlowState> &flow,
                          const std::uint32_t input, const Type output_type) {
  if (flow == nullptr || !flow->status || input == 0u ||
      input > flow->values.size()) {
    return 0u;
  }
  const Type input_type = flow->values[input - 1u].type;
  if (!valid_type(input_type) || !valid_type(output_type)) {
    reject(*flow, Reason::GraphTypeMismatch);
    return 0u;
  }
  if (type_fixed(input_type) || type_fixed(output_type)) {
    reject(*flow, Reason::FixedFormatMismatch);
    return 0u;
  }
  if (input_type == output_type) {
    return input;
  }
  return flow_retype_value(flow, input, output_type, {}, false);
}

std::uint32_t flow_retype_like(const std::shared_ptr<FlowState> &flow,
                               const std::uint32_t input,
                               const std::uint32_t format_source) {
  if (flow == nullptr || !flow->status || input == 0u ||
      input > flow->values.size() || format_source == 0u ||
      format_source > flow->values.size()) {
    return 0u;
  }
  const FlowValue &target = flow->values[format_source - 1u];
  const FlowValue &source = flow->values[input - 1u];
  if (!valid_type(source.type) || !valid_type(target.type)) {
    reject(*flow, Reason::GraphTypeMismatch);
    return 0u;
  }
  if (source.type == target.type &&
      source.fixed_format == target.fixed_format) {
    return input;
  }
  return flow_retype_value(flow, input, target.type, target.fixed_format, true);
}

std::uint32_t flow_fixed_select_value(const std::shared_ptr<FlowState> &flow,
                                      const std::uint32_t value,
                                      const std::uint32_t selected,
                                      const std::uint64_t otherwise_bits,
                                      const bool invert,
                                      const std::string_view name) {
  if (flow == nullptr || !flow->status || value == 0u || selected == 0u ||
      value > flow->values.size() || selected > flow->values.size()) {
    return 0u;
  }
  const FlowValue &source = flow->values[value - 1u];
  const FlowValue &selector = flow->values[selected - 1u];
  if (!type_fixed(source.type) || selector.type != source.type ||
      selector.count != source.count ||
      selector.fixed_format != source.fixed_format) {
    reject(*flow, Reason::FixedFormatMismatch);
    return 0u;
  }
  const auto expression = make_expr();
  const ExprRef source_value =
      detail::input(expression, source.type, 0u, source.fixed_format);
  const ExprRef selected_value =
      detail::input(expression, selector.type, 1u, selector.fixed_format);
  const ExprRef zero =
      constant(expression, selector.type, 0u, selector.fixed_format);
  const ExprRef condition =
      binary(invert ? ExprOp::Equal : ExprOp::NotEqual, selected_value, zero);
  const ExprRef otherwise =
      constant(expression, source.type, otherwise_bits, source.fixed_format);
  const ExprRef chosen =
      ternary(ExprOp::Select, condition, source_value, otherwise);
  const ExprRef stored =
      quantize_expr(chosen, source.type, source.fixed_format);
  const std::array inputs{value, selected};
  return flow_map_value(flow, inputs, name, stored);
}

std::uint32_t flow_fixed_merge_values(const std::shared_ptr<FlowState> &flow,
                                      const std::uint32_t left,
                                      const std::uint32_t right,
                                      const Window operation,
                                      const std::string_view name) {
  if (flow == nullptr || !flow->status || left == 0u || right == 0u ||
      left > flow->values.size() || right > flow->values.size()) {
    return 0u;
  }
  const FlowValue &left_value = flow->values[left - 1u];
  const FlowValue &right_value = flow->values[right - 1u];
  if (!type_fixed(left_value.type) || right_value.type != left_value.type ||
      right_value.count != left_value.count ||
      !same_fixed_storage_policy(right_value.fixed_format,
                                 left_value.fixed_format)) {
    reject(*flow, Reason::FixedFormatMismatch);
    return 0u;
  }
  FixedFormat output_format = left_value.fixed_format;
  if (static_cast<unsigned>(right_value.fixed_format.approximation) >
      static_cast<unsigned>(output_format.approximation)) {
    output_format.approximation = right_value.fixed_format.approximation;
  }
  const auto expression = make_expr();
  const ExprRef left_expression =
      detail::input(expression, left_value.type, 0u, left_value.fixed_format);
  const ExprRef right_expression =
      detail::input(expression, right_value.type, 1u, right_value.fixed_format);
  ExprRef merged;
  switch (operation) {
  case Window::Sum:
    merged = binary(ExprOp::Add, left_expression, right_expression);
    break;
  case Window::Min:
    merged = ternary(ExprOp::Select,
                     binary(ExprOp::Less, left_expression, right_expression),
                     left_expression, right_expression);
    break;
  case Window::Max:
    merged = ternary(ExprOp::Select,
                     binary(ExprOp::Greater, left_expression, right_expression),
                     left_expression, right_expression);
    break;
  default:
    reject(*flow, Reason::StencilOpUnsupported);
    return 0u;
  }
  const ExprRef stored = quantize_expr(merged, left_value.type, output_format);
  const std::array inputs{left, right};
  return flow_map_value(flow, inputs, name, stored);
}

void flow_reject(const std::shared_ptr<FlowState> &flow, const Reason reason) {
  if (flow != nullptr) {
    reject(*flow, reason);
  }
}

} // namespace rund::compute::detail
