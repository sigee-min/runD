#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::net {

enum class Family : std::uint8_t {
  None = 0u,
  IPv4 = 4u,
  IPv6 = 6u,
};

enum class Transport : std::uint8_t {
  Stream = 1u,
  Datagram = 2u,
};

class Address final {
public:
  constexpr Address() noexcept = default;

  [[nodiscard]] static constexpr Address
  ipv4(const std::array<std::byte, 4u> bytes,
       const std::uint16_t port = 0u) noexcept {
    Address address;
    address.family_ = Family::IPv4;
    address.port_ = port;
    for (std::size_t index = 0u; index < bytes.size(); ++index) {
      address.bytes_[index] = bytes[index];
    }
    return address;
  }

  [[nodiscard]] static constexpr Address
  ipv6(const std::array<std::byte, 16u> bytes,
       const std::uint16_t port = 0u) noexcept {
    Address address;
    address.family_ = Family::IPv6;
    address.port_ = port;
    address.bytes_ = bytes;
    return address;
  }

  [[nodiscard]] static constexpr Address
  loopback(const Family family, const std::uint16_t port = 0u) noexcept {
    if (family == Family::IPv4) {
      return ipv4({std::byte{127u}, std::byte{0u}, std::byte{0u},
                   std::byte{1u}},
                  port);
    }
    if (family == Family::IPv6) {
      std::array<std::byte, 16u> bytes{};
      bytes.back() = std::byte{1u};
      return ipv6(bytes, port);
    }
    return {};
  }

  [[nodiscard]] constexpr Family family() const noexcept { return family_; }
  [[nodiscard]] constexpr std::uint16_t port() const noexcept { return port_; }
  [[nodiscard]] constexpr std::span<const std::byte> bytes() const noexcept {
    if (family_ == Family::IPv4) {
      return {bytes_.data(), 4u};
    }
    if (family_ == Family::IPv6) {
      return bytes_;
    }
    return {};
  }
  [[nodiscard]] constexpr bool valid() const noexcept {
    return family_ == Family::IPv4 || family_ == Family::IPv6;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }

  friend constexpr bool operator==(const Address &, const Address &) noexcept =
      default;

private:
  Family family_ = Family::None;
  std::array<std::byte, 16u> bytes_{};
  std::uint16_t port_ = 0u;
};

static_assert(sizeof(Address) == 20u);
static_assert(sizeof(Family) == 1u);
static_assert(sizeof(Transport) == 1u);

} // namespace rund::net
