#include "src/accel/context/internal.hpp"

#include <cstddef>
#include <cstdint>

namespace node_accel_contract {
namespace {
using KernelBindingIndices = rund::node::accel::detail::KernelBindingIndices;

[[nodiscard]] bool PushRange(KernelBindingIndices& indices,
                             const std::uint64_t base,
                             const std::uint64_t count) {
  for (std::uint64_t index = 0u; index < count; ++index) {
    if (!indices.push_back(base + index)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool ContainsRange(const KernelBindingIndices& indices,
                                 const std::uint64_t base,
                                 const std::size_t count) {
  for (std::size_t index = 0u; index < count; ++index) {
    if (indices[index] != base + static_cast<std::uint64_t>(index)) {
      return false;
    }
  }
  return true;
}

}  // namespace

[[nodiscard]] bool KernelBindingIndicesUseInlineStorageUntilOverflow() {
  KernelBindingIndices inline_indices{};
  if (!PushRange(inline_indices, 10u, 4u) || !inline_indices.valid() ||
      inline_indices.heap || inline_indices.size() != 4u ||
      !inline_indices.overflow_indices.empty() ||
      !ContainsRange(inline_indices, 10u, 4u)) {
    return false;
  }

  KernelBindingIndices overflow_indices{};
  if (!PushRange(overflow_indices, 20u, 5u) || !overflow_indices.valid() ||
      !overflow_indices.heap || overflow_indices.size() != 5u ||
      overflow_indices.overflow_indices.size() != 5u ||
      !ContainsRange(overflow_indices, 20u, 5u)) {
    return false;
  }

  KernelBindingIndices reserved_indices{};
  reserved_indices.reserve(5u);
  return PushRange(reserved_indices, 30u, 5u) &&
         reserved_indices.valid() && reserved_indices.heap &&
         reserved_indices.size() == 5u && reserved_indices[4] == 34u;
}

}  // namespace node_accel_contract
