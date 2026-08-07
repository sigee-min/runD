#include "model.hpp"

#include "../../type.hpp"

#include <algorithm>
#include <array>

namespace rund::compute::detail {
namespace {

struct ExpressionStateValidation final {
  const ExprState *state = nullptr;
  std::array<FixedFormat, MaxMapInputs> fixed_inputs{};
  std::uint16_t fixed_input_mask = 0u;
  FixedFormat fallback{};
  bool fixed_inputs_consistent = true;
  bool fallback_consistent = true;
};

[[nodiscard]] bool
expression_state_ok(const ExprState &state, const std::span<const Type> inputs,
                    InputFixedFormats &input_formats,
                    ExpressionStateValidation &validation) noexcept {
  for (std::size_t index = 0; index < state.nodes.size(); ++index) {
    const ExprNode &node = state.nodes[index];
    if (!supported(node.operation) || !valid_type(node.type)) {
      return false;
    }
    const auto valid_ref = [index](const std::uint32_t ref) {
      return ref > 0 && ref <= index;
    };
    if (node.operation == ExprOp::Input) {
      if (node.left >= inputs.size() || node.type != inputs[node.left]) {
        return false;
      }
      if (type_fixed(node.type)) {
        const auto bit = static_cast<std::uint16_t>(
            std::uint16_t{1u} << static_cast<std::size_t>(node.left));
        if ((validation.fixed_input_mask & bit) == 0u) {
          validation.fixed_inputs[node.left] = node.fixed_format;
          validation.fixed_input_mask |= bit;
        } else if (validation.fixed_inputs[node.left] != node.fixed_format) {
          validation.fixed_inputs_consistent = false;
        }
        if (node.fixed_format.integer_bits != 0u &&
            (input_formats.present & bit) == 0u) {
          input_formats.values[node.left] = node.fixed_format;
          input_formats.present |= bit;
        }
      }
      continue;
    }
    const std::uint8_t arity = expr_arity(node.operation);
    if (arity == InvalidArity || (arity >= 1u && !valid_ref(node.left)) ||
        (arity >= 2u && !valid_ref(node.right)) ||
        (arity == 3u && !valid_ref(node.third))) {
      return false;
    }
  }
  return true;
}

} // namespace

bool expressions_ok(const std::span<const Type> outputs,
                    const std::span<const Type> inputs,
                    const std::span<const ExprRef> expressions,
                    InputFixedFormats &input_formats) noexcept {
  if (outputs.empty() || outputs.size() != expressions.size() ||
      outputs.size() > MaxOutputs || inputs.size() > MaxMapInputs) {
    return false;
  }
  if (!std::all_of(outputs.begin(), outputs.end(), valid_type) ||
      !std::all_of(inputs.begin(), inputs.end(), valid_type)) {
    return false;
  }
  const std::size_t scalar_bytes = type_bytes(outputs.front());
  if (scalar_bytes == 0u) {
    return false;
  }
  const bool width_mask = outputs.size() == 1u && !inputs.empty() &&
                          is_width_mask(expressions.front(), inputs.front());
  std::array<ExpressionStateValidation, MaxOutputs> state_validations{};
  std::size_t state_count = 0u;
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    const ExprRef &expression = expressions[index];
    if (expression.type != outputs[index] || expression.state == nullptr ||
        !expression.state->status || expression.node == 0u ||
        expression.node > expression.state->nodes.size() ||
        !valid_type(expression.type) ||
        type_bytes(outputs[index]) != scalar_bytes) {
      return false;
    }
    std::size_t state_index = 0u;
    for (; state_index < state_count; ++state_index) {
      if (state_validations[state_index].state == expression.state.get()) {
        break;
      }
    }
    if (state_index == state_count) {
      ExpressionStateValidation &validation = state_validations[state_count++];
      validation.state = expression.state.get();
      validation.fallback = expression.fixed_format;
      if (!expression_state_ok(*expression.state, inputs, input_formats,
                               validation)) {
        return false;
      }
    } else if (state_validations[state_index].fallback !=
               expression.fixed_format) {
      state_validations[state_index].fallback_consistent = false;
    }
    if (type_fixed(outputs[index])) {
      const ExprOp root =
          expression.state->nodes[expression.node - 1u].operation;
      if (root != ExprOp::Input && root != ExprOp::Quantize &&
          root != ExprOp::BoundaryMask) {
        return false;
      }
    }
  }
  for (std::size_t state_index = 0u; state_index < state_count; ++state_index) {
    const ExpressionStateValidation &validation =
        state_validations[state_index];
    if (!validation.fixed_inputs_consistent) {
      return false;
    }
    for (std::size_t input = 0u; input < inputs.size(); ++input) {
      const auto bit = static_cast<std::uint16_t>(std::uint16_t{1u} << input);
      if ((validation.fixed_input_mask & bit) == 0u) {
        continue;
      }
      if ((input_formats.present & bit) == 0u &&
          !validation.fallback_consistent) {
        return false;
      }
      if (validation.fixed_inputs[input] !=
          input_formats.get(input, validation.fallback)) {
        return false;
      }
    }
  }
  for (const Type input : inputs) {
    if (type_bytes(input) != scalar_bytes && !width_mask) {
      return false;
    }
  }
  return true;
}

bool fixed_output_missing_quantize(
    const std::span<const Type> outputs,
    const std::span<const ExprRef> expressions) noexcept {
  if (outputs.size() != expressions.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < outputs.size(); ++index) {
    if (!type_fixed(outputs[index])) {
      continue;
    }
    const ExprRef &expression = expressions[index];
    if (expression.state == nullptr || expression.node == 0u ||
        expression.node > expression.state->nodes.size()) {
      continue;
    }
    const ExprOp root = expression.state->nodes[expression.node - 1u].operation;
    if (root != ExprOp::Input && root != ExprOp::Quantize &&
        root != ExprOp::BoundaryMask) {
      return true;
    }
  }
  return false;
}

} // namespace rund::compute::detail
