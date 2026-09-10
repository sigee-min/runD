#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

[[nodiscard]] PreparedAffineRun BuildAffinePlan(const PreparedRun &prepared,
                                                const u64 scalar_bytes) {
  if (prepared.domain == rund::kernel::ComputeDomain::Fixed ||
      prepared.read_count != 1u || prepared.write_count != 1u ||
      prepared.instructions.empty()) {
    return {};
  }
  struct Form final {
    u64 a{}, b{};
    bool valid{};
  };
  std::vector<Form> forms(prepared.value_slot_count);
  const auto get = [&](u32 index) {
    return index < forms.size() ? forms[index] : Form{};
  };
  for (std::size_t index = 0u; index < prepared.instructions.size(); ++index) {
    const auto &instruction = prepared.instructions[index];
    const auto &node = instruction.node;
    const auto op = static_cast<IrOp>(node.op);
    Form value{};
    switch (op) {
    case IrOp::Param:
      value = {0u, instruction.immediate, true};
      break;
    case IrOp::Constant:
      value = {
          0u,
          static_cast<u64>(node.lhs) |
              (scalar_bytes == 8u ? static_cast<u64>(node.rhs) << 32u : 0u),
          true};
      break;
    case IrOp::Index:
      // Public Map authoring may retain an unused lane index. Its normal
      // invocation overflow check still runs; any consumer declines affine.
      value = {};
      break;
    case IrOp::Read:
      if (instruction.binding_slot() != 0u)
        return {};
      value = {1u, 0u, true};
      break;
    case IrOp::Add:
    case IrOp::Sub:
    case IrOp::Mul:
    case IrOp::MulWrap: {
      const Form left = get(node.lhs), right = get(node.rhs);
      if (!left.valid || !right.valid)
        return {};
      if (op == IrOp::Add)
        value = {left.a + right.a, left.b + right.b, true};
      else if (op == IrOp::Sub)
        value = {left.a - right.a, left.b - right.b, true};
      else {
        if (left.a != 0u && right.a != 0u)
          return {};
        value = {left.a * right.b + right.a * left.b, left.b * right.b, true};
      }
      break;
    }
    case IrOp::Write: {
      const Form output = get(node.lhs);
      if (!output.valid || index + 1u != prepared.instructions.size() ||
          instruction.binding_slot() != 0u ||
          instruction.element_bytes != scalar_bytes)
        return {};
      return {output.a, output.b, true};
    }
    default:
      return {};
    }
    if (instruction.value_index >= forms.size())
      return {};
    // Snapshot operands before assigning: physical slots may alias last uses.
    forms[instruction.value_index] = value;
  }
  return {};
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
