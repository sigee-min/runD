#include "test/assert.hpp"

#include "local.hpp"

#include <node/runtime/replay/code.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace {

void CheckDefinedCode(const rund::replay::Code code,
                      const rund::replay::Family expected_family,
                      const std::uint16_t expected_value,
                      const std::string_view expected_error) {
  TEST_ASSERT(rund::replay::valid(code));
  TEST_ASSERT(!rund::replay::ok(code));
  TEST_ASSERT(rund::replay::exit_code(code) == 1);
  TEST_ASSERT(rund::replay::family(code) == expected_family);
  TEST_ASSERT((rund::replay::raw(code) & 0xffffu) == expected_value);
  TEST_ASSERT(rund::replay::error(code) == expected_error);
}

void CheckRuntimeCode(const rund::ReasonCode reason,
                      const rund::replay::Code expected,
                      const std::uint16_t expected_value,
                      const std::string_view expected_error) {
  const rund::replay::Code code = rund::node::replay_detail::code(reason);
  TEST_ASSERT(code == expected);
  TEST_ASSERT(rund::replay::valid(code));
  TEST_ASSERT((rund::replay::raw(code) & 0xffffu) == expected_value);
  TEST_ASSERT(rund::node::replay_detail::reason(code) == reason);
  if (reason == rund::ReasonCode::Ok) {
    TEST_ASSERT(rund::replay::family(code) == rund::replay::Family::None);
    TEST_ASSERT(rund::replay::error(code).empty());
    TEST_ASSERT(rund::replay::exit_code(code) == 0);
  } else {
    TEST_ASSERT(rund::replay::family(code) == rund::replay::Family::Runtime);
    TEST_ASSERT(rund::replay::error(code) == expected_error);
    TEST_ASSERT(rund::replay::exit_code(code) == 1);
  }
}

void CheckCodeContract() {
  static_assert(rund::replay::ok(rund::replay::Code::Ok));
  static_assert(rund::replay::exit_code(rund::replay::Code::Ok) == 0);
  TEST_ASSERT(rund::replay::valid(rund::replay::Code::Ok));
  TEST_ASSERT(rund::replay::family(rund::replay::Code::Ok) ==
              rund::replay::Family::None);
  TEST_ASSERT(rund::replay::error(rund::replay::Code::Ok).empty());

#define RUND_REPLAY_CODE(domain, value, name, text)                            \
  CheckDefinedCode(rund::replay::Code::name, rund::replay::Family::domain,     \
                   value, text);
#include <rund/replay/code.def>
#undef RUND_REPLAY_CODE

#define RUND_NODE_REASON(value, name, text, category)                          \
  CheckRuntimeCode(rund::ReasonCode::name, rund::replay::Code::name, value,    \
                   text);
#include <rund/reason.def>
#undef RUND_NODE_REASON
}

void CheckHeaderAndCanonicalIntegerRejection(
    const std::vector<std::byte> &canonical) {
  TEST_ASSERT(canonical.size() > 18u);

  std::vector<std::byte> bad_magic = canonical;
  bad_magic.front() ^= std::byte{0xffu};
  const auto magic = rund::node::DecodeRuntimeReplayRecord(bad_magic);
  TEST_ASSERT(!magic);
  TEST_ASSERT(magic.code == rund::replay::Code::CodecBadHeader);

  std::vector<std::byte> bad_schema = canonical;
  bad_schema[6] ^= std::byte{1u};
  TEST_ASSERT(!rund::node::DecodeRuntimeReplayRecord(bad_schema));

  std::vector<std::byte> bad_kind = canonical;
  bad_kind[7] ^= std::byte{1u};
  TEST_ASSERT(!rund::node::DecodeRuntimeReplayRecord(bad_kind));

  // The fixture outcome is Ok, whose canonical varuint is one zero byte.
  std::vector<std::byte> overlong = canonical;
  overlong[8] = std::byte{0x80u};
  overlong.insert(overlong.begin() + 9, std::byte{0u});
  const auto noncanonical = rund::node::DecodeRuntimeReplayRecord(overlong);
  TEST_ASSERT(!noncanonical);
  TEST_ASSERT(noncanonical.code == rund::replay::Code::CodecBadField);
}

void CheckIntegrityAndExactConsumption(
    const std::vector<std::byte> &canonical) {
  // Header (8), Ok code (1), and start hash (8) precede the saved checksum.
  std::vector<std::byte> bad_checksum = canonical;
  bad_checksum[17] ^= std::byte{1u};
  const auto checksum =
      rund::node::DecodeRuntimeReplayRecord(bad_checksum);
  TEST_ASSERT(!checksum);
  TEST_ASSERT(checksum.code == rund::replay::Code::CodecHashInvalid);

  std::vector<std::byte> trailing = canonical;
  trailing.push_back(std::byte{0u});
  TEST_ASSERT(!rund::node::DecodeRuntimeReplayRecord(trailing));

  std::vector<std::byte> zero_start = canonical;
  for (std::size_t index = 9u; index < 17u; ++index) {
    zero_start.at(index) = std::byte{0u};
  }
  const auto start = rund::node::DecodeRuntimeReplayRecord(zero_start);
  TEST_ASSERT(!start);
  TEST_ASSERT(start.code == rund::replay::Code::CodecBadField);

  for (std::size_t cut : {std::size_t{0u}, std::size_t{1u},
                          std::size_t{7u}, std::size_t{8u},
                          canonical.size() / 2u, canonical.size() - 1u}) {
    TEST_ASSERT(
        !rund::node::DecodeRuntimeReplayRecord(
            std::span<const std::byte>{canonical}.first(cut)));
  }
}

void CheckAdmission(const std::vector<std::byte> &canonical) {
  rund::replay::Limits bytes{};
  bytes.max_bytes = canonical.size() - 1u;
  const auto encoded =
      rund::replay::Record::load(ReplayArtifact(canonical), bytes);
  TEST_ASSERT(!encoded);
  TEST_ASSERT(encoded.code() ==
              rund::replay::Code::CodecEncodedCapacityExceeded);

  rund::replay::Limits entries{};
  entries.max_entries = 0u;
  const auto aggregate =
      rund::node::DecodeRuntimeReplayRecord(canonical, entries);
  TEST_ASSERT(!aggregate);
  TEST_ASSERT(aggregate.code ==
              rund::replay::Code::CodecEntryCapacityExceeded);
}

} // namespace

int CheckReplayDecodeContract(const RuntimeReplayFixture &fixture) {
  CheckCodeContract();
  CheckHeaderAndCanonicalIntegerRejection(fixture.encoded);
  CheckIntegrityAndExactConsumption(fixture.encoded);
  CheckAdmission(fixture.encoded);

  const auto first = rund::node::DecodeRuntimeReplayRecord(fixture.encoded);
  const auto second = rund::node::DecodeRuntimeReplayRecord(fixture.encoded);
  TEST_ASSERT(first.ok());
  TEST_ASSERT(second.ok());
  TEST_ASSERT(SaveReplayRecord(first.record) == fixture.encoded);
  TEST_ASSERT(SaveReplayRecord(second.record) == fixture.encoded);
  return 0;
}
