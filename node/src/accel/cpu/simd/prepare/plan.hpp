#pragma once

#include <kernel/program/compute/lowering/resource.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace rund::node::accel::cpu_simd_detail {
namespace {

inline constexpr u32 kNoValueSlot = std::numeric_limits<u32>::max();

[[nodiscard]] bool StableReference(const std::vector<std::uint8_t> &stable,
                                   const u32 value) noexcept {
  return value == 0u || (value < stable.size() && stable[value] != 0u);
}

[[nodiscard]] bool
StableNode(const IrOp op, const ParsedNode &node,
           const std::vector<std::uint8_t> &stable) noexcept {
  if (op == IrOp::Param || op == IrOp::Constant || op == IrOp::ReadUniform) {
    return true;
  }
  if (op == IrOp::Read || op == IrOp::ReadAt || op == IrOp::Write ||
      op == IrOp::Index) {
    return false;
  }
  const rund::kernel::compute_lowering_detail::ParsedNodeResources resources =
      rund::kernel::compute_lowering_detail::ParsedNodeResourcesFor(node);
  if (!resources.ok || !resources.produces_value) {
    return false;
  }
  for (u32 index = 0u; index < resources.ref_count; ++index) {
    if (!StableReference(stable, resources.refs[index])) {
      return false;
    }
  }
  return true;
}

struct ValueReferences final {
  std::array<u32, 3u> values{};
  std::size_t count = 0u;
};

inline void AppendValueReference(ValueReferences &references,
                                 const u32 value) noexcept {
  if (value == 0u) {
    return;
  }
  for (std::size_t index = 0u; index < references.count; ++index) {
    if (references.values[index] == value) {
      return;
    }
  }
  references.values[references.count++] = value;
}

[[nodiscard]] ValueReferences
NodeValueReferences(const ParsedNode &node) noexcept {
  ValueReferences references{};
  const rund::kernel::compute_lowering_detail::ParsedNodeResources resources =
      rund::kernel::compute_lowering_detail::ParsedNodeResourcesFor(node);
  if (!resources.ok) {
    return references;
  }
  for (u32 index = 0u; index < resources.ref_count; ++index) {
    AppendValueReference(references, resources.refs[index]);
  }
  return references;
}

[[nodiscard]] bool
RemapNodeValueReferences(ParsedNode &node,
                         const std::vector<u32> &slot_by_value) noexcept {
  const rund::kernel::compute_lowering_detail::ParsedNodeResources resources =
      rund::kernel::compute_lowering_detail::ParsedNodeResourcesFor(node);
  if (!resources.ok) {
    return false;
  }
  std::array<u32 *, 3u> operands{&node.lhs, &node.rhs, &node.aux};
  for (u32 index = 0u; index < resources.ref_count; ++index) {
    u32 &value = *operands[index];
    if (value == 0u || value >= slot_by_value.size() ||
        value != resources.refs[index] ||
        slot_by_value[value] == kNoValueSlot) {
      return false;
    }
    value = slot_by_value[value];
  }
  return true;
}

[[nodiscard]] PreparedInstruction
PrepareInstruction(const ParsedIR &parsed, const BindingPlan &bindings,
                   const rund::kernel::BindingSet &runtime,
                   const u64 scalar_bytes,
                   const std::size_t node_index) noexcept {
  const ParsedNode &node = parsed.nodes[node_index];
  const IrOp op = static_cast<IrOp>(node.op);
  const CpuSimdExecutorSlot base_executor = CpuSimdBaseExecutorSlot(node.op);
  PreparedInstruction instruction{
      .node = node,
      .value_index = static_cast<u32>(node_index + 1u),
      .full_executor_slot = base_executor,
      .tail_executor_slot = base_executor,
  };
  const rund::kernel::compute_lowering_detail::ParsedNodeResources resources =
      rund::kernel::compute_lowering_detail::ParsedNodeResourcesFor(node);
  for (u32 index = 0u; index < resources.ref_count; ++index) {
    const u32 value = resources.refs[index];
    if (value != 0u && value <= parsed.nodes.size()) {
      instruction.set_operand_fraction(
          index, parsed.nodes[value - 1u].fixed_format.fraction_bits);
    }
  }
  if (op == IrOp::Param) {
    const ParsedBinding &binding = parsed.bindings[node.aux];
    std::memcpy(&instruction.immediate, binding.value_bytes.data(),
                static_cast<std::size_t>(binding.element_bytes));
  } else if (op == IrOp::Read) {
    instruction.set_binding_slot(bindings.slots[node.aux]);
    const rund::kernel::BufferSpan &span =
        runtime.input_buffers[instruction.binding_slot()];
    instruction.full_executor_slot = span.stride_bytes == scalar_bytes
                                         ? kCpuSimdReadFullExecutorSlot
                                         : kCpuSimdReadStridedFullExecutorSlot;
  } else if (op == IrOp::ReadUniform) {
    instruction.set_binding_slot(bindings.slots[node.aux]);
  } else if (op == IrOp::ReadAt) {
    instruction.set_binding_slot(bindings.slots[node.aux]);
    instruction.immediate = bindings.slots[node.lhs];
    instruction.element_bytes = node.rhs;
  } else if (op == IrOp::Write) {
    instruction.set_binding_slot(bindings.slots[node.aux]);
    instruction.element_bytes = parsed.bindings[node.aux].element_bytes;
    if (runtime.output_buffer_count != 0u) {
      const rund::kernel::OutputSpan &span =
          runtime.output_buffers[instruction.binding_slot()];
      if (span.element_bytes == scalar_bytes &&
          span.stride_bytes == scalar_bytes) {
        instruction.full_executor_slot = kCpuSimdWriteFullExecutorSlot;
      }
    } else if (runtime.output_bytes_per_tile == scalar_bytes &&
               runtime.staged_output_stride == scalar_bytes) {
      instruction.full_executor_slot = kCpuSimdWriteFullExecutorSlot;
    }
  }
  return instruction;
}

[[nodiscard]] const char *
BuildPreparedPlan(const ParsedIR &parsed, const BindingPlan &binding_plan,
                  const rund::kernel::BindingSet &bindings,
                  const u64 scalar_bytes, PreparedRun &prepared) {
  const std::size_t node_count = parsed.nodes.size();
  std::vector<std::uint8_t> stable(node_count + 1u, std::uint8_t{0u});
  std::size_t once_count = 0u;
  u64 write_count = 0u;
  for (std::size_t index = 0u; index < node_count; ++index) {
    const ParsedNode &node = parsed.nodes[index];
    const IrOp op = static_cast<IrOp>(node.op);
    const bool is_stable = StableNode(op, node, stable);
    stable[index + 1u] = is_stable ? std::uint8_t{1u} : std::uint8_t{0u};
    once_count += is_stable ? 1u : 0u;
    write_count += op == IrOp::Write ? 1u : 0u;
    prepared.uses_index = prepared.uses_index || op == IrOp::Index;
  }
  if (write_count == 0u || write_count != binding_plan.write_count) {
    return "cpu_simd_instruction_plan_invalid";
  }
  prepared.read_count = static_cast<u32>(binding_plan.read_count);
  prepared.write_count = static_cast<u32>(binding_plan.write_count);

  std::vector<std::size_t> execution_order;
  execution_order.reserve(node_count);
  for (const bool stable_group : {true, false}) {
    for (std::size_t index = 0u; index < node_count; ++index) {
      if ((stable[index + 1u] != 0u) == stable_group) {
        execution_order.push_back(index);
      }
    }
  }
  if (execution_order.size() != node_count) {
    return "cpu_simd_instruction_plan_invalid";
  }

  std::vector<std::size_t> last_use(node_count + 1u, 0u);
  std::vector<std::uint8_t> pinned(node_count + 1u, std::uint8_t{0u});
  for (std::size_t position = 0u; position < execution_order.size();
       ++position) {
    const std::size_t node_index = execution_order[position];
    last_use[node_index + 1u] = position;
    const ValueReferences references =
        NodeValueReferences(parsed.nodes[node_index]);
    for (std::size_t ref_index = 0u; ref_index < references.count;
         ++ref_index) {
      const u32 value = references.values[ref_index];
      if (value >= last_use.size()) {
        return "cpu_simd_instruction_plan_invalid";
      }
      last_use[value] = std::max(last_use[value], position);
      if (position >= once_count && stable[value] != 0u) {
        pinned[value] = std::uint8_t{1u};
      }
    }
  }
  for (std::size_t value = 1u; value < pinned.size(); ++value) {
    if (pinned[value] != 0u) {
      last_use[value] = execution_order.size();
    }
  }

  prepared.instructions.reserve(node_count);
  prepared.value_slot_count = 0u;
  std::vector<u32> slot_by_value(node_count + 1u, kNoValueSlot);
  std::vector<u32> free_slots;
  const auto release = [&](const u32 value) {
    const u32 slot = slot_by_value[value];
    if (slot == kNoValueSlot) {
      return false;
    }
    free_slots.push_back(slot);
    slot_by_value[value] = kNoValueSlot;
    return true;
  };
  for (std::size_t position = 0u; position < execution_order.size();
       ++position) {
    const std::size_t node_index = execution_order[position];
    const ParsedNode &source_node = parsed.nodes[node_index];
    const IrOp op = static_cast<IrOp>(source_node.op);
    PreparedInstruction instruction = PrepareInstruction(
        parsed, binding_plan, bindings, scalar_bytes, node_index);
    if (!CpuSimdExecutorSlotValid(instruction.full_executor_slot) ||
        !CpuSimdExecutorSlotValid(instruction.tail_executor_slot) ||
        !RemapNodeValueReferences(instruction.node, slot_by_value)) {
      return "cpu_simd_instruction_plan_invalid";
    }
    const u32 value = static_cast<u32>(node_index + 1u);
    if (op == IrOp::Write) {
      instruction.value_index = 0u;
    } else {
      const ValueReferences references = NodeValueReferences(source_node);
      u32 aliased_value = 0u;
      for (std::size_t ref_index = 0u; ref_index < references.count;
           ++ref_index) {
        const u32 reference = references.values[ref_index];
        if (last_use[reference] == position && pinned[reference] == 0u) {
          aliased_value = reference;
          break;
        }
      }

      u32 slot = kNoValueSlot;
      if (aliased_value != 0u) {
        slot = slot_by_value[aliased_value];
        slot_by_value[aliased_value] = kNoValueSlot;
      } else if (!free_slots.empty()) {
        slot = free_slots.back();
        free_slots.pop_back();
      } else {
        if (prepared.value_slot_count >
            static_cast<std::size_t>(std::numeric_limits<u32>::max())) {
          return "cpu_simd_instruction_plan_invalid";
        }
        slot = static_cast<u32>(prepared.value_slot_count++);
      }
      if (slot == kNoValueSlot) {
        return "cpu_simd_instruction_plan_invalid";
      }
      slot_by_value[value] = slot;
      instruction.value_index = slot;
    }
    prepared.instructions.push_back(std::move(instruction));

    const ValueReferences references = NodeValueReferences(source_node);
    for (std::size_t ref_index = 0u; ref_index < references.count;
         ++ref_index) {
      const u32 reference = references.values[ref_index];
      if (last_use[reference] == position && pinned[reference] == 0u &&
          slot_by_value[reference] != kNoValueSlot && !release(reference)) {
        return "cpu_simd_instruction_plan_invalid";
      }
    }
    if (op != IrOp::Write && last_use[value] == position &&
        pinned[value] == 0u && !release(value)) {
      return "cpu_simd_instruction_plan_invalid";
    }
  }
  prepared.once_count = once_count;
  return prepared.instructions.size() == node_count &&
                 prepared.value_slot_count != 0u
             ? nullptr
             : "cpu_simd_instruction_plan_invalid";
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
