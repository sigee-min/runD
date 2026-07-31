#pragma once

#include <cstring>
#include <utility>
#include <vector>

namespace rund::node::accel::cpu_simd_detail {
namespace {

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
      op == IrOp::Index || op == IrOp::Quantize) {
    return false;
  }
  return StableReference(stable, node.lhs) &&
         StableReference(stable, node.rhs) && StableReference(stable, node.aux);
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
  if (op == IrOp::Param) {
    const ParsedBinding &binding = parsed.bindings[node.aux];
    std::memcpy(&instruction.immediate, binding.value_bytes.data(),
                static_cast<std::size_t>(binding.element_bytes));
  } else if (op == IrOp::Read) {
    instruction.binding_slot = bindings.slots[node.aux];
    const rund::kernel::BufferSpan &span =
        runtime.input_buffers[instruction.binding_slot];
    instruction.full_executor_slot = span.stride_bytes == scalar_bytes
                                         ? kCpuSimdReadFullExecutorSlot
                                         : kCpuSimdReadStridedFullExecutorSlot;
  } else if (op == IrOp::ReadUniform) {
    instruction.binding_slot = bindings.slots[node.aux];
  } else if (op == IrOp::ReadAt) {
    instruction.binding_slot = bindings.slots[node.aux];
    instruction.immediate = bindings.slots[node.lhs];
    instruction.element_bytes = node.rhs;
  } else if (op == IrOp::Write) {
    instruction.binding_slot = bindings.slots[node.aux];
    instruction.element_bytes = parsed.bindings[node.aux].element_bytes;
    if (runtime.output_buffer_count != 0u) {
      const rund::kernel::OutputSpan &span =
          runtime.output_buffers[instruction.binding_slot];
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
  prepared.value_formats.resize(node_count + 1u);
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
    prepared.value_formats[index + 1u] = node.fixed_format;
  }
  if (write_count == 0u || write_count != binding_plan.write_count) {
    return "cpu_simd_instruction_plan_invalid";
  }
  prepared.read_count = static_cast<u32>(binding_plan.read_count);
  prepared.write_count = static_cast<u32>(binding_plan.write_count);

  prepared.instructions.reserve(node_count);
  const auto append = [&](const bool stable_group) {
    for (std::size_t index = 0u; index < node_count; ++index) {
      if ((stable[index + 1u] != 0u) != stable_group) {
        continue;
      }
      PreparedInstruction instruction = PrepareInstruction(
          parsed, binding_plan, bindings, scalar_bytes, index);
      if (!CpuSimdExecutorSlotValid(instruction.full_executor_slot) ||
          !CpuSimdExecutorSlotValid(instruction.tail_executor_slot)) {
        return false;
      }
      prepared.instructions.push_back(std::move(instruction));
    }
    return true;
  };
  if (!append(true) || !append(false)) {
    return "cpu_simd_instruction_plan_invalid";
  }
  prepared.once_count = once_count;
  return prepared.instructions.size() == node_count
             ? nullptr
             : "cpu_simd_instruction_plan_invalid";
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
