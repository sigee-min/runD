#include "filter.hpp"
#include "../../expression/state.hpp"
#include "../../type.hpp"
#include "../recipe.hpp"
#include <array>
#include <limits>
#include <memory>
#include <new>
#include <rund/compute/abi/expression.hpp>
#include <utility>

namespace rund::compute::detail {

std::optional<std::uint32_t>
fold_filter_sum(const std::shared_ptr<FlowState> &flow,
                const std::uint32_t values, const std::uint32_t count,
                const Reduce operation) {
  if (operation != Reduce::Sum || values == 0u || count == 0u ||
      values > flow->values.size() || count > flow->values.size()) {
    return std::nullopt;
  }
  const Type type = flow->values[values - 1u].type;
  if (type != Type::U32 && type != Type::U64)
    return std::nullopt;
  std::optional<FilterStep> selected;
  for (auto step = flow->steps.rbegin(); step != flow->steps.rend(); ++step) {
    if (const auto *filter = std::get_if<FilterStep>(&*step);
        filter != nullptr && filter->values == values &&
        filter->count == count) {
      selected = *filter;
      break;
    }
  }
  if (!selected)
    return std::nullopt;
  const FilterStep filter = *selected;
  // Partition offsets and the narrow selected count have a U32 domain. Keep
  // the ordinary Filter path outside it; dead count materialization must not
  // erase a possible count/offset overflow.
  if (flow->values[filter.input - 1u].count >
      std::numeric_limits<std::uint32_t>::max()) {
    return std::nullopt;
  }
  const Type mask_type = flow->values[filter.selected - 1u].type;
  if ((mask_type != Type::U32 && mask_type != Type::U64) ||
      type_bytes(mask_type) != type_bytes(type))
    return std::nullopt;
  const auto expressions = make_expr();
  const ExprRef value = detail::input(expressions, type, 0u);
  const ExprRef mask = detail::input(expressions, mask_type, 1u);
  const ExprRef predicate =
      binary(ExprOp::NotEqual, mask, constant(expressions, mask_type, 0u));
  const ExprRef mapped = ternary(ExprOp::Select, predicate, value,
                                 constant(expressions, type, 0u));
  const std::array inputs{filter.input, filter.selected};
  const std::uint32_t masked =
      flow_map_value(flow, inputs, "filter-sum", mapped);
  if (masked == 0u)
    return std::uint32_t{0u};
  return flow_unary_value(flow, masked, Primitive::Reduce, type, 1u,
                          {.mode = static_cast<std::uint32_t>(Reduce::Sum)});
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
  try {
    flow->steps.emplace_back(
        FilterStep{input, selected, rejected, values, count});
  } catch (const std::bad_alloc &) {
    reject(*flow, Reason::FlowCapacity);
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
