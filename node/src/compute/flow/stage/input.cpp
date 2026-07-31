#include "../recipe.hpp"

#include "../../type.hpp"

#include <memory>

namespace rund::compute::detail {

namespace {

enum class FlowInputRole : unsigned char { Independent, Side };

[[nodiscard]] std::uint32_t
admit_flow_input(const std::shared_ptr<FlowState> &flow, const HostView input,
                 const bool bind, const FixedFormat fixed_format,
                 const FlowInputRole role) {
  if (flow == nullptr || !flow->status) {
    return 0u;
  }
  if (bind && input.data == nullptr && input.count != 0u) {
    reject(*flow, Reason::ShapeMismatch);
    return 0u;
  }
  const bool fixed_input =
      input.type == Type::FixedLane32 || input.type == Type::FixedLane64;
  const bool format_present =
      fixed_format.integer_bits != 0u || fixed_format.fraction_bits != 0u;
  const std::size_t format_width =
      static_cast<std::size_t>(fixed_format.integer_bits) +
      fixed_format.fraction_bits;
  if ((fixed_input && (!format_present || fixed_format.integer_bits == 0u ||
                       fixed_format.fraction_bits == 0u ||
                       format_width != type_bytes(input.type) * 8u)) ||
      (!fixed_input && format_present)) {
    reject(*flow, Reason::FixedFormatMismatch);
    return 0u;
  }
  FixedFormat admitted_format = fixed_format;
  if (role == FlowInputRole::Side && fixed_input && flow->output != 0u &&
      flow->output <= flow->values.size()) {
    const FlowValue &current = flow->values[flow->output - 1u];
    if (current.type == input.type) {
      if (current.fixed_format.integer_bits != fixed_format.integer_bits ||
          current.fixed_format.fraction_bits != fixed_format.fraction_bits) {
        reject(*flow, Reason::FixedFormatMismatch);
        return 0u;
      }
      admitted_format.rounding = current.fixed_format.rounding;
      admitted_format.overflow = current.fixed_format.overflow;
    }
  }
  const std::uint32_t value =
      append(*flow, input.type, input.count, admitted_format);
  if (value == 0u) {
    return 0u;
  }
  try {
    flow->inputs.push_back(value);
    if (bind) {
      flow->bindings.push_back(input);
    }
    return value;
  } catch (const std::bad_alloc &) {
    reject(*flow, Reason::FlowCapacity);
    return 0u;
  }
}

} // namespace

std::uint32_t flow_independent_input(const std::shared_ptr<FlowState> &flow,
                                     const HostView input,
                                     const FixedFormat fixed_format) {
  return admit_flow_input(flow, input, false, fixed_format,
                          FlowInputRole::Independent);
}

void flow_mark_bounded_input(const std::shared_ptr<FlowState> &flow,
                             const std::uint32_t count,
                             const std::size_t capacity) noexcept {
  if (flow == nullptr || !flow->status || count == 0u || capacity == 0u ||
      count > flow->values.size()) {
    if (flow != nullptr && flow->status) {
      reject(*flow, Reason::BoundedCountInvalid);
    }
    return;
  }
  try {
    flow->bounded_inputs.push_back(
        BoundedInputSchema{.count = count, .capacity = capacity});
  } catch (const std::bad_alloc &) {
    reject(*flow, Reason::FlowCapacity);
  }
}

std::uint32_t flow_side(const std::shared_ptr<FlowState> &flow,
                        const HostView input, const bool bind,
                        const FixedFormat fixed_format) {
  return admit_flow_input(flow, input, bind, fixed_format, FlowInputRole::Side);
}

} // namespace rund::compute::detail
