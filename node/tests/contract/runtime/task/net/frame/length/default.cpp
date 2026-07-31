#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/frame/length.hpp>

#include <array>
#include <cstddef>
#include <span>

int RunNetFrameLengthDefaultLimitCase() {
  std::array<std::byte, 4> encoded_bytes{};
  const rund::net::frame::Result default_limit_max =
      rund::net::frame::encode_length(64u * 1024u,
                                      std::span<std::byte>{encoded_bytes});
  TEST_ASSERT(default_limit_max.ok());

  const rund::net::frame::Result default_limit_reject =
      rund::net::frame::encode_length(64u * 1024u + 1u,
                                      std::span<std::byte>{encoded_bytes});
  TEST_ASSERT(!default_limit_reject.ok());
  TEST_ASSERT(default_limit_reject.code() ==
              rund::ReasonCode::NetFrameTooLarge);
  TEST_ASSERT(default_limit_reject.bytes == 64u * 1024u + 1u);

  const std::array<std::byte, 4> default_limit_header{
      std::byte{0x00}, std::byte{0x01}, std::byte{0x00}, std::byte{0x00}};
  const rund::net::frame::Result default_limit_decode =
      rund::net::frame::decode_length(
          std::span<const std::byte>{default_limit_header});
  TEST_ASSERT(default_limit_decode.ok());
  TEST_ASSERT(default_limit_decode.bytes == 64u * 1024u);

  const std::array<std::byte, 4> default_limit_large_header{
      std::byte{0x00}, std::byte{0x01}, std::byte{0x00}, std::byte{0x01}};
  const rund::net::frame::Result default_limit_decode_reject =
      rund::net::frame::decode_length(
          std::span<const std::byte>{default_limit_large_header});
  TEST_ASSERT(!default_limit_decode_reject.ok());
  TEST_ASSERT(default_limit_decode_reject.code() ==
              rund::ReasonCode::NetFrameTooLarge);
  TEST_ASSERT(default_limit_decode_reject.bytes == 64u * 1024u + 1u);
  return 0;
}
