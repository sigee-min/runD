#include "local.hpp"

#include <kernel/program/compute/lowering/resource.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace node_accel_contract::vector_plan {

IndependentSlotLowerBound
IndependentPhysicalSlotLowerBound(const rund::kernel::ComputeIR &ir) {
  using rund::kernel::IrOp;
  using rund::kernel::compute_lowering_detail::ParsedNodeResourcesFor;

  const auto parsed = rund::kernel::compute_lowering_detail::ParseComputeIR(ir);
  if (!parsed.ok) {
    return {};
  }
  const std::size_t count = parsed.nodes.size();
  std::vector<bool> stable(count + 1u, false);
  std::size_t once_count = 0u;
  for (std::size_t index = 0u; index < count; ++index) {
    const auto &node = parsed.nodes[index];
    const auto op = static_cast<IrOp>(node.op);
    const auto resources = ParsedNodeResourcesFor(node);
    if (!resources.ok) {
      return {};
    }
    bool is_stable =
        op == IrOp::Param || op == IrOp::Constant || op == IrOp::ReadUniform;
    if (op != IrOp::Read && op != IrOp::ReadAt && op != IrOp::Write &&
        op != IrOp::Index && !is_stable) {
      is_stable = resources.produces_value;
      for (rund::kernel::u32 ref = 0u; ref < resources.ref_count; ++ref) {
        const auto value = resources.refs[ref];
        is_stable = is_stable && value < stable.size() &&
                    (value == 0u || stable[static_cast<std::size_t>(value)]);
      }
    }
    stable[index + 1u] = is_stable;
    once_count += is_stable ? 1u : 0u;
  }

  std::vector<std::size_t> order;
  order.reserve(count);
  for (const bool stable_group : {true, false}) {
    for (std::size_t index = 0u; index < count; ++index) {
      if (stable[index + 1u] == stable_group) {
        order.push_back(index);
      }
    }
  }
  std::vector<std::size_t> last_use(count + 1u, 0u);
  std::vector<bool> pinned(count + 1u, false);
  for (std::size_t position = 0u; position < order.size(); ++position) {
    const std::size_t node_index = order[position];
    last_use[node_index + 1u] = position;
    const auto resources = ParsedNodeResourcesFor(parsed.nodes[node_index]);
    for (rund::kernel::u32 ref = 0u; ref < resources.ref_count; ++ref) {
      const auto value = resources.refs[ref];
      if (value == 0u || value >= last_use.size()) {
        return {};
      }
      last_use[value] = std::max(last_use[value], position);
      pinned[value] =
          pinned[value] || (position >= once_count && stable[value]);
    }
  }
  for (std::size_t value = 1u; value < pinned.size(); ++value) {
    if (pinned[value]) {
      last_use[value] = order.size();
    }
  }

  std::vector<bool> live(count + 1u, false);
  std::size_t live_count = 0u;
  std::size_t peak = 0u;
  for (std::size_t position = 0u; position < order.size(); ++position) {
    const std::size_t node_index = order[position];
    const auto resources = ParsedNodeResourcesFor(parsed.nodes[node_index]);
    std::array<rund::kernel::u32, 3u> unique{};
    std::size_t unique_count = 0u;
    bool dying_operand = false;
    for (rund::kernel::u32 ref = 0u; ref < resources.ref_count; ++ref) {
      const auto value = resources.refs[ref];
      bool duplicate = false;
      for (std::size_t seen = 0u; seen < unique_count; ++seen) {
        duplicate = duplicate || unique[seen] == value;
      }
      if (!duplicate) {
        unique[unique_count++] = value;
        dying_operand =
            dying_operand || (last_use[value] == position && !pinned[value]);
      }
    }
    peak = std::max(
        peak, live_count + static_cast<std::size_t>(resources.produces_value &&
                                                    !dying_operand));
    for (std::size_t ref = 0u; ref < unique_count; ++ref) {
      const auto value = unique[ref];
      if (last_use[value] == position && !pinned[value] && live[value]) {
        live[value] = false;
        --live_count;
      }
    }
    const std::size_t value = node_index + 1u;
    if (resources.produces_value && last_use[value] > position) {
      live[value] = true;
      ++live_count;
    }
  }
  return IndependentSlotLowerBound{
      .peak = peak, .once_count = once_count, .ok = true};
}

} // namespace node_accel_contract::vector_plan
