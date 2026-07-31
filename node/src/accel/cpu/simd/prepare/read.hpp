#pragma once

namespace rund::node::accel::cpu_simd_detail {
namespace {

struct ReadShape final {
  u64 bytes{};
  u64 count{};
};

[[nodiscard]] ReadShape ReadBindingShape(const ParsedIR &parsed,
                                         const u32 binding,
                                         const u64 scalar_bytes,
                                         const u64 tile_count) noexcept {
  ReadShape shape{.bytes = scalar_bytes};
  for (const ParsedNode &node : parsed.nodes) {
    const auto op = static_cast<IrOp>(node.op);
    if (op == IrOp::Read && node.aux == binding) {
      shape.count = tile_count;
    } else if (op == IrOp::ReadUniform && node.aux == binding) {
      shape.count = std::max<u64>(shape.count, 1u);
    } else if (op == IrOp::ReadAt && node.lhs == binding) {
      shape.bytes = sizeof(u32);
      shape.count = tile_count;
    } else if (op == IrOp::ReadAt && node.aux == binding) {
      shape.count = std::max<u64>(shape.count, node.rhs);
    }
  }
  return shape;
}

[[nodiscard]] const char *
ValidateReadBuffers(const ParsedIR &parsed, const BindingPlan &plan,
                    const rund::kernel::BindingSet &bindings,
                    const u64 scalar_bytes) {
  if (bindings.input_buffer_count != plan.read_count) {
    return "cpu_simd_input_count_mismatch";
  }
  if (plan.read_count != 0u && bindings.input_buffers == nullptr) {
    return "cpu_simd_input_missing";
  }
  if (bindings.input_element_byte_count != 0u &&
      bindings.input_element_byte_count != plan.read_count) {
    return "cpu_simd_input_element_bytes_mismatch";
  }

  for (std::size_t index = 0u; index < parsed.bindings.size(); ++index) {
    if (parsed.bindings[index].kind != kReadBindingKind) {
      continue;
    }
    const u64 slot = plan.slots[index];
    const auto &span = bindings.input_buffers[slot];
    const ReadShape shape = ReadBindingShape(parsed, static_cast<u32>(index),
                                             scalar_bytes, bindings.tile_count);
    if (shape.count == 0u || span.data == nullptr ||
        span.element_bytes != shape.bytes || span.stride_bytes < shape.bytes ||
        !FitsSize(span.stride_bytes) || span.count < shape.count) {
      return "cpu_simd_input_invalid";
    }
    if (bindings.input_element_byte_count != 0u &&
        bindings.input_element_bytes[slot] != shape.bytes) {
      return "cpu_simd_input_element_bytes_mismatch";
    }
    u64 offset = 0u;
    if (!CheckedOffset(shape.count - 1u, span.stride_bytes, shape.bytes,
                       offset) ||
        !FitsSize(offset)) {
      return "cpu_simd_input_bounds_overflow";
    }
  }
  return nullptr;
}

} // namespace
} // namespace rund::node::accel::cpu_simd_detail
