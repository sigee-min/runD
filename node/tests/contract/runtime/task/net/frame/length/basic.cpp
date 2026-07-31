#include "test/assert.hpp"

#include "local.hpp"

#include <rund/net/frame/header.hpp>
#include <rund/net/frame/length.hpp>
#include <rund/net/frame/limit.hpp>

#include <array>
#include <cstddef>
#include <span>
#include <string_view>

int RunNetFrameLengthBasicCase() {
  const rund::net::frame::Limit defaults{};
  TEST_ASSERT(defaults.max_bytes == 64u * 1024u);

  const rund::net::frame::Header header{.bytes = 7u};
  TEST_ASSERT(header.bytes == 7u);

  const rund::net::frame::Result not_started{};
  TEST_ASSERT(!not_started);
  TEST_ASSERT(!not_started.ok());
  TEST_ASSERT(std::string_view{not_started.error()} == "task_invalid");
  TEST_ASSERT(not_started.code() == rund::ReasonCode::TaskInvalid);
  TEST_ASSERT(not_started.bytes == 0u);

  std::array<std::byte, 4> encoded_bytes{};
  const rund::net::frame::Limit large_limit{.max_bytes = 0x02000000u};
  const rund::net::frame::Result encoded = rund::net::frame::encode_length(
      0x01020304u, std::span<std::byte>{encoded_bytes}, large_limit);
  TEST_ASSERT(encoded);
  TEST_ASSERT(encoded.ok());
  TEST_ASSERT(encoded.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(encoded.error().empty());
  TEST_ASSERT(encoded.bytes == 0x01020304u);
  TEST_ASSERT(encoded_bytes[0] == std::byte{0x01});
  TEST_ASSERT(encoded_bytes[1] == std::byte{0x02});
  TEST_ASSERT(encoded_bytes[2] == std::byte{0x03});
  TEST_ASSERT(encoded_bytes[3] == std::byte{0x04});

  const rund::net::frame::Result decoded = rund::net::frame::decode_length(
      std::span<const std::byte>{encoded_bytes}, large_limit);
  TEST_ASSERT(decoded);
  TEST_ASSERT(decoded.ok());
  TEST_ASSERT(decoded.code() == rund::ReasonCode::Ok);
  TEST_ASSERT(decoded.error().empty());
  TEST_ASSERT(decoded.bytes == 0x01020304u);
  return 0;
}
