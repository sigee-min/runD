#pragma once

#include "../../../../device/residency/execution/plan.hpp"
#include "../../../state.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>
#include <variant>

namespace rund::compute::detail {

enum class VirtualDeviceVsmEndpoint : std::uint8_t {
  Invalid,
  Resident,
  Staged
};
enum class VirtualDeviceVsmRouteKind : std::uint8_t {
  Direct,
  StagedLoop,
  WindowRing,
  GraphResident,
};

struct VirtualDeviceVsmDirect final {
  std::uint64_t page_count{};
  std::uint64_t frame_capacity{};
  constexpr bool
  operator==(const VirtualDeviceVsmDirect &) const noexcept = default;
};
struct VirtualDeviceVsmStagedLoop final {
  std::uint64_t page_count{};
  std::uint64_t frame_capacity{};
  constexpr bool
  operator==(const VirtualDeviceVsmStagedLoop &) const noexcept = default;
};
struct VirtualDeviceVsmWindowRing final {
  VirtualWindowPreflight window{};
  constexpr bool
  operator==(const VirtualDeviceVsmWindowRing &) const noexcept = default;
};
struct VirtualDeviceVsmGraphResident final {
  std::uint64_t page_count{};
  std::uint64_t frame_capacity{};
  VirtualDeviceVsmEndpoint endpoint{VirtualDeviceVsmEndpoint::Invalid};
  constexpr bool
  operator==(const VirtualDeviceVsmGraphResident &) const noexcept = default;
};

// One selected physical topology, with no mirrored kind, endpoint, page count,
// or default Window payload. The stamp authenticates this value at admission.
class VirtualDeviceVsmRouteProof final {
public:
  using Route =
      std::variant<VirtualDeviceVsmDirect, VirtualDeviceVsmStagedLoop,
                   VirtualDeviceVsmWindowRing, VirtualDeviceVsmGraphResident>;
  constexpr VirtualDeviceVsmRouteProof() noexcept = default;
  constexpr explicit VirtualDeviceVsmRouteProof(Route route,
                                                std::uint64_t hi = 0u,
                                                std::uint64_t lo = 0u) noexcept
      : stamp_hi(hi), stamp_lo(lo), route_(route) {}

  std::uint64_t stamp_hi{};
  std::uint64_t stamp_lo{};

  [[nodiscard]] constexpr const Route &route() const noexcept { return route_; }
  [[nodiscard]] constexpr VirtualDeviceVsmRouteKind kind() const noexcept {
    return std::visit(
        [](const auto &value) noexcept {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, VirtualDeviceVsmDirect>)
            return VirtualDeviceVsmRouteKind::Direct;
          if constexpr (std::is_same_v<T, VirtualDeviceVsmStagedLoop>)
            return VirtualDeviceVsmRouteKind::StagedLoop;
          if constexpr (std::is_same_v<T, VirtualDeviceVsmWindowRing>)
            return VirtualDeviceVsmRouteKind::WindowRing;
          return VirtualDeviceVsmRouteKind::GraphResident;
        },
        route_);
  }
  [[nodiscard]] constexpr const VirtualWindowPreflight *
  window() const noexcept {
    const auto *ring = std::get_if<VirtualDeviceVsmWindowRing>(&route_);
    return ring == nullptr ? nullptr : &ring->window;
  }
  [[nodiscard]] constexpr VirtualDeviceVsmEndpoint endpoint() const noexcept {
    return std::visit(
        [](const auto &value) noexcept {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, VirtualDeviceVsmDirect>)
            return VirtualDeviceVsmEndpoint::Invalid;
          else if constexpr (std::is_same_v<T, VirtualDeviceVsmStagedLoop>)
            return VirtualDeviceVsmEndpoint::Staged;
          else if constexpr (std::is_same_v<T, VirtualDeviceVsmGraphResident>)
            return value.endpoint;
          else {
            switch (value.window.endpoint) {
            case VirtualWindowPreflightEndpoint::Resident:
              return VirtualDeviceVsmEndpoint::Resident;
            case VirtualWindowPreflightEndpoint::Staged:
              return VirtualDeviceVsmEndpoint::Staged;
            default:
              return VirtualDeviceVsmEndpoint::Invalid;
            }
          }
        },
        route_);
  }
  [[nodiscard]] constexpr std::uint64_t page_count() const noexcept {
    return std::visit(
        [](const auto &value) noexcept {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, VirtualDeviceVsmWindowRing>)
            return value.window.page_count;
          else
            return value.page_count;
        },
        route_);
  }
  [[nodiscard]] constexpr std::uint64_t frame_capacity() const noexcept {
    return std::visit(
        [](const auto &value) noexcept {
          using T = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<T, VirtualDeviceVsmWindowRing>)
            return value.window.frame_capacity;
          else
            return value.frame_capacity;
        },
        route_);
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    if ((stamp_hi == 0u && stamp_lo == 0u) || page_count() < 2u ||
        page_count() > std::numeric_limits<std::uint32_t>::max() ||
        frame_capacity() == 0u || frame_capacity() > page_count())
      return false;
    if (const auto *ring = window()) {
      const VirtualWindowPreflight &window = *ring;
      if (window.mode != VirtualWindowPreflightMode::WindowRing ||
          window.frame_capacity != 2u || window.input_bytes == 0u ||
          window.output_bytes == 0u || window.input_frame_bytes == 0u ||
          window.output_frame_bytes == 0u || window.frame_bytes == 0u ||
          window.device_storage_bytes == 0u || window.host.input == 0u ||
          window.host.output == 0u || window.host.storage_bytes == 0u ||
          window.host.input != window.page_count ||
          window.host.output != window.frame_capacity) {
        return false;
      }
      if (window.endpoint != VirtualWindowPreflightEndpoint::Resident &&
          window.endpoint != VirtualWindowPreflightEndpoint::Staged) {
        return false;
      }
      constexpr std::uint64_t max = std::numeric_limits<std::uint64_t>::max();
      if (window.input_frame_bytes > max - window.output_frame_bytes) {
        return false;
      }
      const std::uint64_t frame_bytes =
          window.input_frame_bytes + window.output_frame_bytes;
      if (frame_bytes != window.frame_bytes ||
          frame_bytes > max / window.frame_capacity) {
        return false;
      }
      const std::uint64_t device_bank_bytes =
          frame_bytes * window.frame_capacity;
      if (device_bank_bytes > max / residency::execution::BankCapacity ||
          window.device_storage_bytes !=
              device_bank_bytes * residency::execution::BankCapacity ||
          window.host.input > max / window.input_frame_bytes ||
          window.host.output > max / window.output_frame_bytes) {
        return false;
      }
      const std::uint64_t host_input_bytes =
          window.host.input * window.input_frame_bytes;
      const std::uint64_t host_output_bytes =
          window.host.output * window.output_frame_bytes;
      if (host_input_bytes > max - host_output_bytes ||
          host_input_bytes + host_output_bytes >
              max / residency::execution::BankCapacity ||
          window.host.storage_bytes != (host_input_bytes + host_output_bytes) *
                                           residency::execution::BankCapacity) {
        return false;
      }
    }
    return kind() != VirtualDeviceVsmRouteKind::GraphResident ||
           endpoint() == VirtualDeviceVsmEndpoint::Resident ||
           endpoint() == VirtualDeviceVsmEndpoint::Staged;
  }
  [[nodiscard]] constexpr bool
  operator==(const VirtualDeviceVsmRouteProof &) const noexcept = default;

private:
  Route route_{};
};

static_assert(sizeof(VirtualDeviceVsmRouteProof) <= 120u);
static_assert(std::is_nothrow_copy_constructible_v<VirtualDeviceVsmRouteProof>);

} // namespace rund::compute::detail
