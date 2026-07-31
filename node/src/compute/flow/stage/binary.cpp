#include "../recipe.hpp"

#include <array>
#include <memory>
#include <span>

namespace rund::compute::detail {

std::uint32_t flow_binary_values(const std::shared_ptr<FlowState> &flow,
                                 const Primitive operation,
                                 const std::span<const std::uint32_t> inputs,
                                 const Type output_type,
                                 const std::size_t count,
                                 const PrimitiveOptions options) {
  if (flow == nullptr || !flow->status || inputs.empty()) {
    return 0u;
  }
  for (const std::uint32_t input : inputs) {
    if (input == 0u || input > flow->values.size()) {
      return 0u;
    }
  }
  FixedFormat fixed_format{};
  bool fixed_format_set = false;
  if (is_fixed(output_type)) {
    for (const std::uint32_t input : inputs) {
      const FlowValue &source = flow->values[input - 1u];
      if (source.type != output_type) {
        continue;
      }
      if (!fixed_format_set) {
        fixed_format = source.fixed_format;
        fixed_format_set = true;
      } else if (!same_fixed_storage_policy(source.fixed_format,
                                            fixed_format)) {
        reject(*flow, Reason::FixedFormatMismatch);
        return 0u;
      } else if (static_cast<unsigned>(source.fixed_format.approximation) >
                 static_cast<unsigned>(fixed_format.approximation)) {
        fixed_format.approximation = source.fixed_format.approximation;
      }
    }
  }
  const std::uint32_t output = append(*flow, output_type, count, fixed_format);
  if (output == 0u) {
    return 0u;
  }
  if (operation == Primitive::Gather && inputs.size() == 3u) {
    flow->values[output - 1u].active = inputs[2u];
  }
  const std::array outputs{output};
  return append_primitive(*flow, inputs, outputs, operation, options) ? output
                                                                      : 0u;
}

void flow_binary(const std::shared_ptr<FlowState> &flow,
                 const Primitive operation, const std::uint32_t side,
                 const bool side_first, const Type output_type,
                 const std::size_t count, const PrimitiveOptions options) {
  if (flow == nullptr || !flow->status) {
    return;
  }
  const std::array inputs = side_first ? std::array{side, flow->output}
                                       : std::array{flow->output, side};
  const std::uint32_t output =
      flow_binary_values(flow, operation, inputs, output_type, count, options);
  if (output != 0u) {
    flow->output = output;
  }
}

std::uint32_t flow_complex_side(const std::shared_ptr<FlowState> &flow,
                                const HostView input, const bool bind,
                                const FixedFormat fixed_format) {
  if (flow == nullptr || !flow->status || flow->output == 0u) {
    return 0u;
  }
  const FlowValue &real = flow->values[flow->output - 1u];
  if (input.type != real.type || input.count != real.count) {
    reject(*flow, Reason::TransformShapeMismatch);
    return 0u;
  }
  if (is_fixed(real.type) &&
      (fixed_format.integer_bits != real.fixed_format.integer_bits ||
       fixed_format.fraction_bits != real.fixed_format.fraction_bits)) {
    reject(*flow, Reason::FixedFormatMismatch);
    return 0u;
  }
  return flow_side(flow, input, bind, fixed_format);
}

ComplexIds flow_transform(const std::shared_ptr<FlowState> &flow,
                          const std::uint32_t real, const std::uint32_t imag,
                          const PrimitiveOptions options) {
  if (flow == nullptr || !flow->status || real == 0u || imag == 0u ||
      real > flow->values.size() || imag > flow->values.size()) {
    return {};
  }
  const FlowValue value = flow->values[real - 1u];
  if (flow->values[imag - 1u].type != value.type ||
      flow->values[imag - 1u].count != value.count ||
      (is_fixed(value.type) &&
       !same_fixed_storage_policy(flow->values[imag - 1u].fixed_format,
                                  value.fixed_format))) {
    reject(*flow, Reason::TransformShapeMismatch);
    return {};
  }
  FixedFormat fixed_format = value.fixed_format;
  if (is_fixed(value.type)) {
    fixed_format.approximation = Approximation::Deterministic;
  }
  const std::uint32_t output_real =
      append(*flow, value.type, value.count, fixed_format);
  const std::uint32_t output_imag =
      append(*flow, value.type, value.count, fixed_format);
  if (output_real == 0u || output_imag == 0u) {
    return {};
  }
  const std::array inputs{real, imag};
  const std::array outputs{output_real, output_imag};
  if (!append_primitive(*flow, inputs, outputs, Primitive::Transform,
                        options)) {
    return {};
  }
  flow->output = output_real;
  return ComplexIds{output_real, output_imag};
}

} // namespace rund::compute::detail
