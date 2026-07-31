#pragma once

#include <accel/device.hpp>

#include "run/execute.hpp"

namespace node_accel_contract::collective {

template <typename Key, std::size_t Count>
[[nodiscard]] bool SortBackendsAgree(const rund::AccelDevice &metal,
                                     const rund::AccelDevice &vulkan,
                                     const rund::kernel::ComputeScalar scalar,
                                     const std::array<Key, Count> &input_keys) {
  const SortRunHashes<Key> metal_hash =
      SortHashesMatchCpuReference(metal, scalar, input_keys);
  const SortRunHashes<Key> vulkan_hash =
      SortHashesMatchCpuReference(vulkan, scalar, input_keys);
  return metal_hash.ok && vulkan_hash.ok &&
         metal_hash.key_hash == vulkan_hash.key_hash &&
         metal_hash.value_hash == vulkan_hash.value_hash;
}

} // namespace node_accel_contract::collective
