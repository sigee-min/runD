#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include <memory>

namespace rund::node::accel::detail {

[[nodiscard]] bool SameReason(const char *lhs, const char *rhs) noexcept;

template <typename Left, typename Right>
[[nodiscard]] bool SameOwner(const std::shared_ptr<Left> &lhs,
                             const std::shared_ptr<Right> &rhs) noexcept {
  const std::owner_less<void> less{};
  return lhs != nullptr && rhs != nullptr && !less(lhs, rhs) && !less(rhs, lhs);
}

template <typename Left, typename Right>
[[nodiscard]] bool SameObject(const std::shared_ptr<Left> &lhs,
                              const std::shared_ptr<Right> &rhs) noexcept {
  return static_cast<const void *>(lhs.get()) ==
             static_cast<const void *>(rhs.get()) &&
         SameOwner(lhs, rhs);
}

template <typename Value>
[[nodiscard]] bool SameOwner(const std::weak_ptr<Value> &lhs,
                             const std::shared_ptr<void> &rhs) noexcept {
  const std::owner_less<void> less{};
  return !less(lhs, rhs) && !less(rhs, lhs);
}

[[nodiscard]] bool SameCheck(const rund::AccelCheck &lhs,
                             const rund::AccelCheck &rhs) noexcept;

[[nodiscard]] bool SameCaps(const rund::kernel::ComputeCaps &lhs,
                            const rund::kernel::ComputeCaps &rhs) noexcept;

[[nodiscard]] bool SameCpuCaps(const rund::kernel::CpuCaps &lhs,
                               const rund::kernel::CpuCaps &rhs) noexcept;

[[nodiscard]] bool
SameDispatch(const rund::kernel::ComputeBackendDispatch &lhs,
             const rund::kernel::ComputeBackendDispatch &rhs) noexcept;

[[nodiscard]] bool SameBackendInfo(const rund::AccelBackendInfo &lhs,
                                   const rund::AccelBackendInfo &rhs) noexcept;

} // namespace rund::node::accel::detail
