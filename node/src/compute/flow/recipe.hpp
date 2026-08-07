#pragma once

#include "state.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rund::compute::detail {

[[nodiscard]] constexpr bool
same_fixed_storage_policy(const FixedFormat left,
                          const FixedFormat right) noexcept {
  return left.integer_bits == right.integer_bits &&
         left.fraction_bits == right.fraction_bits &&
         left.rounding == right.rounding && left.overflow == right.overflow;
}

inline void reject(FlowState &flow, const Reason reason) noexcept {
  if (flow.status) {
    flow.status = Status::fail(reason);
  }
}

[[nodiscard]] inline std::uint32_t append(FlowState &flow, const Type type,
                                          const std::size_t count,
                                          const FixedFormat fixed_format = {}) {
  try {
    flow.values.push_back(
        FlowValue{.type = type, .fixed_format = fixed_format, .count = count});
    return static_cast<std::uint32_t>(flow.values.size());
  } catch (const std::bad_alloc &) {
    reject(flow, Reason::FlowCapacity);
    return 0u;
  }
}

[[nodiscard]] inline bool
append_primitive(FlowState &flow, const std::span<const std::uint32_t> inputs,
                 const std::span<const std::uint32_t> outputs,
                 const Primitive operation, const PrimitiveOptions options,
                 const FlowControl control = {}) {
  if (inputs.empty()) {
    reject(flow, Reason::GraphBindingInvalid);
    return false;
  }
  try {
    const std::optional<ValueRoutes> routes =
        flow.value_ids.store(inputs, outputs);
    if (!routes) {
      reject(flow, Reason::FlowCapacity);
      return false;
    }
    flow.steps.emplace_back(FlowPrimitive{
        .inputs = routes->inputs,
        .outputs = routes->outputs,
        .operation = operation,
        .options = options,
        .control = control,
    });
    return true;
  } catch (const std::bad_alloc &) {
    reject(flow, Reason::FlowCapacity);
    return false;
  }
}

[[nodiscard]] inline bool
append_map(FlowState &flow, const std::string_view name,
           const std::span<const std::uint32_t> inputs,
           const std::span<const std::uint32_t> outputs,
           const std::span<const ExprRef> expressions,
           const FlowControl control = {}) {
  try {
    const std::optional<ValueRoutes> routes =
        flow.value_ids.store(inputs, outputs);
    if (!routes) {
      reject(flow, Reason::FlowCapacity);
      return false;
    }
    flow.steps.emplace_back(MapStep{
        .name = std::string{name},
        .inputs = routes->inputs,
        .outputs = routes->outputs,
        .expressions =
            std::vector<ExprRef>{expressions.begin(), expressions.end()},
        .control = control,
    });
    return true;
  } catch (const std::bad_alloc &) {
    reject(flow, Reason::FlowCapacity);
    return false;
  }
}

[[nodiscard]] inline Result<std::shared_ptr<ProgramState>>
fail_flow(const Reason reason) {
  return Result<std::shared_ptr<ProgramState>>::fail(reason);
}

} // namespace rund::compute::detail
