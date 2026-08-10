#pragma once

#include <cstdint>
#include <kernel/program/compute/binding/model.hpp>
#include <memory>

namespace rund::node::accel::detail {

struct MetalAdapter;

struct MetalResidentOwner final {
  MetalAdapter *adapter = nullptr;
  std::shared_ptr<void> adapter_owner{};
  std::shared_ptr<void> buffer{};
  rund::kernel::ResidentBufferRef ref{};
  std::uint64_t id = 0u;

  MetalResidentOwner() = default;
  MetalResidentOwner(const MetalResidentOwner &) = delete;
  MetalResidentOwner &operator=(const MetalResidentOwner &) = delete;
  ~MetalResidentOwner();
};

// The resident handle crosses the common backend boundary as shared_ptr<void>.
// Keep its concrete type as an authenticated capability in the control block;
// callers must recover it through LookupMetalResidentOwner rather than an
// unchecked void-pointer cast.
struct MetalResidentOwnerDelete final {
  MetalResidentOwner *owner = nullptr;

  void operator()(MetalResidentOwner *value) const noexcept;
};

[[nodiscard]] std::shared_ptr<MetalResidentOwner>
MakeMetalResidentOwner() noexcept;

[[nodiscard]] std::shared_ptr<MetalResidentOwner>
LookupMetalResidentOwner(const std::shared_ptr<void> &owner) noexcept;

} // namespace rund::node::accel::detail
