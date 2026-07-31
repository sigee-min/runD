#pragma once

#include <rund/compute/backend.hpp>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace rund::compute {

template <class T> class Result;

namespace detail {
struct DeviceState;
using OpenTarget =
    Result<std::shared_ptr<DeviceState>> (*)(std::uint32_t workers);
[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_cpu(std::uint32_t workers);
[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_metal(std::uint32_t workers);
[[nodiscard]] Result<std::shared_ptr<DeviceState>>
open_vulkan(std::uint32_t workers);
struct TargetAccess;
} // namespace detail

class Target final {
public:
  [[nodiscard]] static constexpr Target
  cpu(const std::uint32_t workers = 0u) noexcept {
    return Target{Backend::Cpu, workers, &detail::open_cpu};
  }

  [[nodiscard]] static constexpr Target metal() noexcept {
    return Target{Backend::Metal, 0u, &detail::open_metal};
  }

  [[nodiscard]] static constexpr Target vulkan() noexcept {
    return Target{Backend::Vulkan, 0u, &detail::open_vulkan};
  }

  [[nodiscard]] constexpr Backend backend() const noexcept { return backend_; }
  [[nodiscard]] constexpr std::uint32_t workers() const noexcept {
    return workers_;
  }

  friend constexpr bool operator==(const Target &,
                                   const Target &) noexcept = default;

private:
  friend struct detail::TargetAccess;

  constexpr Target(const Backend backend, const std::uint32_t workers,
                   const detail::OpenTarget open) noexcept
      : open_(open), workers_(workers), backend_(backend) {}

  detail::OpenTarget open_{};
  std::uint32_t workers_{};
  Backend backend_{Backend::Cpu};
};

namespace detail {
struct TargetAccess final {
  [[nodiscard]] static constexpr OpenTarget open(const Target target) noexcept {
    return target.open_;
  }
};
} // namespace detail

static_assert(std::is_trivially_copyable_v<Target>);
static_assert(sizeof(Target) == 16u);

} // namespace rund::compute
