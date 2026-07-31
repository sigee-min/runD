#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/node.hpp>

#include "../bindings.hpp"

#include "../local.hpp"

#include <cstdint>

namespace rund::node::accel::detail {

namespace {

[[nodiscard]] bool
ValueTypesEqual(const rund::kernel::GraphValueType &lhs,
                const rund::kernel::GraphValueType &rhs) noexcept {
  return lhs.kind == rhs.kind && lhs.role == rhs.role &&
         lhs.element_bytes == rhs.element_bytes && lhs.count == rhs.count &&
         lhs.rows == rhs.rows && lhs.cols == rhs.cols &&
         lhs.batch_count == rhs.batch_count;
}

[[nodiscard]] bool
SignaturesEqual(const rund::kernel::GraphSignature &lhs,
                const rund::kernel::GraphSignature &rhs) noexcept {
  if (!lhs.ok || !rhs.ok || lhs.kind != rhs.kind ||
      lhs.value_count != rhs.value_count ||
      lhs.output_count != rhs.output_count ||
      lhs.status_count != rhs.status_count) {
    return false;
  }
  for (rund::kernel::u64 index = 0u; index < lhs.value_count; ++index) {
    if (!ValueTypesEqual(lhs.values[index], rhs.values[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ElementBytesMatch(const std::uint64_t actual,
                                     const std::uint32_t expected) noexcept {
  return expected == 0u ? SortKeyWidthOk(actual) : actual == expected;
}

[[nodiscard]] bool CountMatches(const std::uint64_t actual,
                                const std::uint64_t expected,
                                const rund::kernel::BufferRole role) noexcept {
  if (expected == 0u) {
    return true;
  }
  return role == rund::kernel::BufferRole::Read ? actual >= expected
                                                : actual == expected;
}

[[nodiscard]] bool
BuffersMatchSignature(const rund::AccelGraphNode &node,
                      const rund::kernel::GraphSignature &signature) noexcept {
  if (!signature.ok || node.buffers == nullptr ||
      node.buffer_count < signature.value_count ||
      !node.control.valid(node.buffer_count)) {
    return false;
  }
  for (std::uint64_t index = 0u; index < signature.value_count; ++index) {
    const rund::AccelGraphBufferRef &ref = node.buffers[index];
    const rund::kernel::GraphValueType &value = signature.values[index];
    const bool write = value.role == rund::kernel::BufferRole::Write;
    if (!AccelGraphBufferShapeValid(ref) || ref.role != value.role ||
        !ElementBytesMatch(AccelGraphBufferShape(ref).scalar_width_bytes,
                           value.element_bytes) ||
        !CountMatches(AccelGraphBufferShape(ref).count, value.count,
                      value.role) ||
        !CollectiveUsageOk(AccelGraphBufferShape(ref).usage, write)) {
      return false;
    }
  }
  return true;
}

} // namespace

bool NodeSignatureOk(const rund::AccelGraphNode &node,
                     const rund::kernel::GraphSignature &expected) noexcept {
  return node.signature.ok && SignaturesEqual(node.signature, expected) &&
         BuffersMatchSignature(node, expected);
}

} // namespace rund::node::accel::detail
