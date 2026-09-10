#pragma once

#include "../../../kernel/backend/template/arithmetic.hpp"

#include <cstdint>
#include <vector>

namespace rund::node::accel::detail {

[[nodiscard]] inline bool
AddVulkanHostBytes(std::uint64_t &target, const std::uint64_t count,
                   const std::uint64_t element) noexcept {
  std::uint64_t bytes = 0u;
  return backend_template_plan::product(count, element, bytes) &&
         backend_template_plan::add(target, bytes);
}

template <class T>
[[nodiscard]] inline bool
AddVulkanVectorStorage(std::uint64_t &target,
                       const std::vector<T> &values) noexcept {
  return AddVulkanHostBytes(
      target, static_cast<std::uint64_t>(values.capacity()), sizeof(T));
}

} // namespace rund::node::accel::detail
