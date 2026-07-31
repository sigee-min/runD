#pragma once

#include <rund/net/frame/limit.hpp>
#include <rund/net/frame/result/length.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::net::frame {

namespace length_detail {

[[nodiscard]] constexpr Result reject(const std::uint32_t bytes,
                                      const ::rund::ReasonCode code) noexcept {
  Result result{code};
  result.bytes = bytes;
  return result;
}

[[nodiscard]] constexpr Result accept(const std::uint32_t bytes) noexcept {
  Result result{::rund::ReasonCode::Ok};
  result.bytes = bytes;
  return result;
}

[[nodiscard]] constexpr std::uint32_t
decode_big_endian_u32(const std::span<const std::byte> in) noexcept {
  return (std::to_integer<std::uint32_t>(in[0]) << 24u) |
         (std::to_integer<std::uint32_t>(in[1]) << 16u) |
         (std::to_integer<std::uint32_t>(in[2]) << 8u) |
         std::to_integer<std::uint32_t>(in[3]);
}

} // namespace length_detail

[[nodiscard]] constexpr Result encode_length(const std::uint32_t bytes,
                                             const std::span<std::byte> out,
                                             const Limit limit = {}) noexcept {
  if (out.size() < 4u) {
    return length_detail::reject(bytes, ::rund::ReasonCode::NetFrameHeaderTooSmall);
  }
  if (bytes > limit.max_bytes) {
    return length_detail::reject(bytes, ::rund::ReasonCode::NetFrameTooLarge);
  }

  out[0] = static_cast<std::byte>((bytes >> 24u) & 0xffu);
  out[1] = static_cast<std::byte>((bytes >> 16u) & 0xffu);
  out[2] = static_cast<std::byte>((bytes >> 8u) & 0xffu);
  out[3] = static_cast<std::byte>(bytes & 0xffu);
  return length_detail::accept(bytes);
}

[[nodiscard]] constexpr Result
decode_length(const std::span<const std::byte> in,
              const Limit limit = {}) noexcept {
  if (in.size() < 4u) {
    return length_detail::reject(0u, ::rund::ReasonCode::NetFrameHeaderTooSmall);
  }

  const std::uint32_t bytes = length_detail::decode_big_endian_u32(in);
  if (bytes > limit.max_bytes) {
    return length_detail::reject(bytes, ::rund::ReasonCode::NetFrameTooLarge);
  }
  return length_detail::accept(bytes);
}

} // namespace rund::net::frame
