#pragma once

#include <cstdint>
#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;

struct MetalResidentOwner final {
  MetalAdapter *adapter = nullptr;
  std::shared_ptr<void> adapter_owner{};
  std::shared_ptr<void> buffer{};
  std::uint64_t id = 0u;

  MetalResidentOwner() = default;
  MetalResidentOwner(const MetalResidentOwner &) = delete;
  MetalResidentOwner &operator=(const MetalResidentOwner &) = delete;
  ~MetalResidentOwner();
};

} // namespace rund::node::accel::detail
