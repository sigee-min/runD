#pragma once

#include <rund/compute/abi/state.hpp>
#include <rund/compute/fixed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::compute::detail {

struct HostView final {
  const void *data{};
  std::size_t count{};
  Type type{Type::I32};
};
enum class ResourceAccess : unsigned char { Read, Write };
struct ResourceView final {
  std::shared_ptr<BufferState> buffer;
  Type type{Type::I32};
  FixedFormat format{};
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
  ResourceAccess access{ResourceAccess::Read};
};

} // namespace rund::compute::detail
