#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/frame/length.hpp>
#include <rund/net/frame/limit.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <string_view>

int RunNetFrameLengthRejectCase() {
  std::array<std::byte, 4> encoded_bytes{};
  const rund::net::frame::Limit large_limit{.max_bytes = 0x02000000u};
  std::array<std::byte, 3> short_header{};
  const rund::net::frame::Result short_encode = rund::net::frame::encode_length(
      1u, std::span<std::byte>{short_header}, large_limit);
  TEST_ASSERT(!short_encode);
  TEST_ASSERT(!short_encode.ok());
  TEST_ASSERT(short_encode.code() == rund::ReasonCode::NetFrameHeaderTooSmall);
  TEST_ASSERT(std::string_view{short_encode.error()} ==
              "net_frame_header_too_small");

  const rund::net::frame::Result short_decode = rund::net::frame::decode_length(
      std::span<const std::byte>{short_header}, large_limit);
  TEST_ASSERT(!short_decode);
  TEST_ASSERT(!short_decode.ok());
  TEST_ASSERT(short_decode.code() == rund::ReasonCode::NetFrameHeaderTooSmall);
  TEST_ASSERT(std::string_view{short_decode.error()} ==
              "net_frame_header_too_small");

  const rund::net::frame::Result oversized_encode =
      rund::net::frame::encode_length(
          0x02000001u, std::span<std::byte>{encoded_bytes}, large_limit);
  TEST_ASSERT(!oversized_encode.ok());
  TEST_ASSERT(oversized_encode.code() == rund::ReasonCode::NetFrameTooLarge);
  TEST_ASSERT(std::string_view{oversized_encode.error()} ==
              "net_frame_too_large");
  TEST_ASSERT(oversized_encode.bytes == 0x02000001u);

  const std::array<std::byte, 4> oversized_header{
      std::byte{0x02}, std::byte{0x00}, std::byte{0x00}, std::byte{0x01}};
  const rund::net::frame::Result oversized_decode =
      rund::net::frame::decode_length(
          std::span<const std::byte>{oversized_header}, large_limit);
  TEST_ASSERT(!oversized_decode.ok());
  TEST_ASSERT(oversized_decode.code() == rund::ReasonCode::NetFrameTooLarge);
  TEST_ASSERT(std::string_view{oversized_decode.error()} ==
              "net_frame_too_large");
  TEST_ASSERT(oversized_decode.bytes == 0x02000001u);
  return 0;
}
