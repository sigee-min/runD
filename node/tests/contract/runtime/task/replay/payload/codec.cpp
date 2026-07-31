#include "test/assert.hpp"

#include "../../coroutine/allocation.hpp"
#include "src/runtime/replay/host/payload/codec.hpp"
#include "src/runtime/replay/host/payload/hash.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace {

using rund::node::replay_detail::payload::AppendDecodedBytes;
using rund::node::replay_detail::payload::ByteHash;
using rund::node::replay_detail::payload::Codec;
using rund::node::replay_detail::payload::DecodeRle;
using rund::node::replay_detail::payload::EncodeRle;
using rund::node::replay_detail::payload::Verify;

int RleRoundTripsRepeatedBytes() {
  std::vector<std::byte> bytes{};
  bytes.insert(bytes.end(), 140u, std::byte{0x7b});
  bytes.push_back(std::byte{0x01});
  bytes.push_back(std::byte{0x02});
  bytes.push_back(std::byte{0x03});
  bytes.insert(bytes.end(), 4u, std::byte{0x44});

  const std::vector<std::byte> encoded = EncodeRle(bytes);
  const std::optional<std::vector<std::byte>> decoded =
      DecodeRle(encoded, bytes.size());
  TEST_ASSERT(decoded.has_value());
  TEST_ASSERT(decoded.value() == bytes);
  return 0;
}

int RleRejectsTruncatedRepeatRun() {
  const std::vector<std::byte> encoded{std::byte{0x80}};
  TEST_ASSERT(!DecodeRle(encoded, 3u).has_value());
  return 0;
}

int RleRejectsExpansionMismatch() {
  const std::vector<std::byte> encoded{std::byte{0x80}, std::byte{'x'}};
  TEST_ASSERT(!DecodeRle(encoded, 4u).has_value());
  TEST_ASSERT(!DecodeRle(encoded, 2u).has_value());
  return 0;
}

int VerificationUsesNoDecodedVector() {
  std::vector<std::byte> bytes(140u, std::byte{0x7b});
  const auto hash = rund::host::hash_bytes(bytes.data(), bytes.size());
  std::vector<std::byte> rle = EncodeRle(bytes);

  runtime_task_allocation::Start();
  const bool raw_valid = Verify(hash, bytes.size(), Codec::Raw, bytes);
  const bool rle_valid = Verify(hash, bytes.size(), Codec::Rle, rle);
  runtime_task_allocation::Stop();
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(raw_valid);
  TEST_ASSERT(rle_valid);

  rle.back() ^= std::byte{0x01};
  TEST_ASSERT(!Verify(hash, bytes.size(), Codec::Rle, rle));
  TEST_ASSERT(!Verify(hash, bytes.size() + 1u, Codec::Raw, bytes));
  return 0;
}

int AggregateHashingUsesNoDecodedVector() {
  const std::vector<std::byte> raw{std::byte{0x10}, std::byte{0x20},
                                   std::byte{0x30}};
  const std::vector<std::byte> repeated(140u, std::byte{0x7b});
  const std::vector<std::byte> rle = EncodeRle(repeated);
  std::vector<std::byte> expected = raw;
  expected.insert(expected.end(), repeated.begin(), repeated.end());
  const auto expected_hash =
      rund::host::hash_bytes(expected.data(), expected.size());

  ByteHash aggregate{};
  runtime_task_allocation::Start();
  const bool raw_valid =
      AppendDecodedBytes(raw.size(), Codec::Raw, raw, aggregate);
  const bool rle_valid =
      AppendDecodedBytes(repeated.size(), Codec::Rle, rle, aggregate);
  runtime_task_allocation::Stop();
  TEST_ASSERT(runtime_task_allocation::Count() == 0u);
  TEST_ASSERT(raw_valid);
  TEST_ASSERT(rle_valid);
  TEST_ASSERT(aggregate.Finish() == expected_hash.value);

  ByteHash malformed{};
  TEST_ASSERT(!AppendDecodedBytes(raw.size() + 1u, Codec::Raw, raw, malformed));
  const std::vector<std::byte> truncated{std::byte{0x80}};
  TEST_ASSERT(!AppendDecodedBytes(3u, Codec::Rle, truncated, malformed));
  return 0;
}

} // namespace

int CheckReplayArtifactContract();

int RunRuntimeTaskReplayPayloadCodecContract() {
  if (const int rc = CheckReplayArtifactContract(); rc != 0) {
    return rc;
  }
  if (const int rc = RleRoundTripsRepeatedBytes(); rc != 0) {
    return rc;
  }
  if (const int rc = RleRejectsTruncatedRepeatRun(); rc != 0) {
    return rc;
  }
  if (const int rc = RleRejectsExpansionMismatch(); rc != 0) {
    return rc;
  }
  if (const int rc = VerificationUsesNoDecodedVector(); rc != 0) {
    return rc;
  }
  return AggregateHashingUsesNoDecodedVector();
}
