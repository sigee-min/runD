#include "local.hpp"

#include <rund/net/frame/length.hpp>
#include <rund/net/frame/limit.hpp>

#include <array>
#include <cstddef>
#include <span>

namespace {

constexpr bool CompileTimeFrameLengthCodec() {
  constexpr rund::net::frame::Limit limit{.max_bytes = 0x02000000u};
  std::array<std::byte, 4> out{};
  const rund::net::frame::Result encoded = rund::net::frame::encode_length(
      0x01020304u, std::span<std::byte>{out}, limit);
  if (!encoded.ok() || encoded.code() != rund::ReasonCode::Ok ||
      encoded.bytes != 0x01020304u) {
    return false;
  }
  if (out[0] != std::byte{0x01} || out[1] != std::byte{0x02} ||
      out[2] != std::byte{0x03} || out[3] != std::byte{0x04}) {
    return false;
  }

  const rund::net::frame::Result decoded =
      rund::net::frame::decode_length(std::span<const std::byte>{out}, limit);
  if (!decoded.ok() || decoded.code() != rund::ReasonCode::Ok ||
      decoded.bytes != 0x01020304u) {
    return false;
  }

  std::array<std::byte, 3> too_small{};
  const rund::net::frame::Result small_encode = rund::net::frame::encode_length(
      1u, std::span<std::byte>{too_small}, limit);
  if (small_encode.ok() ||
      small_encode.code() != rund::ReasonCode::NetFrameHeaderTooSmall) {
    return false;
  }
  const rund::net::frame::Result small_decode = rund::net::frame::decode_length(
      std::span<const std::byte>{too_small}, limit);
  if (small_decode.ok() ||
      small_decode.code() != rund::ReasonCode::NetFrameHeaderTooSmall) {
    return false;
  }

  const rund::net::frame::Result too_large = rund::net::frame::encode_length(
      9u, std::span<std::byte>{out}, rund::net::frame::Limit{.max_bytes = 8u});
  return !too_large.ok() &&
         too_large.code() == rund::ReasonCode::NetFrameTooLarge &&
         too_large.bytes == 9u;
}

static_assert(CompileTimeFrameLengthCodec());

} // namespace

int RunNetFrameLengthCompileCase() { return 0; }
