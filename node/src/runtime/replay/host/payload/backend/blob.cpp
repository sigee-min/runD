#include "../backend.hpp"
#include "../codec.hpp"
#include "../hash.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <utility>

namespace rund::node::replay_detail::payload {

Blob Encode(const std::span<const std::byte> bytes,
            const ::rund::StableHash payload_hash) {
  std::vector<std::byte> rle = EncodeRle(bytes);
  const bool use_rle = rle.size() < bytes.size();
  std::vector<std::byte> encoded =
      use_rle ? std::move(rle)
              : std::vector<std::byte>{bytes.begin(), bytes.end()};
  const std::uint64_t encoded_size = static_cast<std::uint64_t>(encoded.size());
  return Blob{
      .payload_hash = payload_hash,
      .uncompressed_bytes = static_cast<std::uint64_t>(bytes.size()),
      .encoded_bytes = encoded_size,
      .codec = use_rle ? Codec::Rle : Codec::Raw,
      .encoded = ::rund::node::replay_detail::payload::Bytes::freeze(
          std::move(encoded)),
  };
}

ReadResult Decode(const ::rund::StableHash payload_hash,
                  const std::uint64_t uncompressed_bytes, const Codec codec,
                  const std::span<const std::byte> encoded) {
  if (uncompressed_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return ReadResult{.code = ::rund::replay::Code::HostPayloadHashInvalid,
                      .bytes = {}};
  }
  std::vector<std::byte> decoded(static_cast<std::size_t>(uncompressed_bytes));
  const ReadStatus status = DecodeInto(payload_hash, uncompressed_bytes, codec,
                                       encoded, decoded, nullptr);
  if (!status.ok()) {
    return ReadResult{.code = status.code, .bytes = {}};
  }
  return ReadResult{.code = ::rund::replay::Code::Ok,
                    .bytes = std::move(decoded)};
}

ReadStatus DecodeInto(const ::rund::StableHash payload_hash,
                      const std::uint64_t uncompressed_bytes, const Codec codec,
                      const std::span<const std::byte> encoded,
                      const std::span<std::byte> output,
                      ByteHash *const record_hash) noexcept {
  if (uncompressed_bytes != output.size()) {
    return ReadStatus{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  ByteHash chunk_hash{};
  bool decoded = false;
  switch (codec) {
  case Codec::Raw:
    decoded = encoded.size() == output.size();
    if (decoded) {
      std::copy(encoded.begin(), encoded.end(), output.begin());
      chunk_hash.Append(encoded);
      if (record_hash != nullptr) {
        record_hash->Append(encoded);
      }
    }
    break;
  case Codec::Rle:
    decoded = DecodeRleInto(encoded, output, &chunk_hash, record_hash);
    break;
  }
  if (!decoded || chunk_hash.Finish() != payload_hash.value) {
    return ReadStatus{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  return ReadStatus{.code = ::rund::replay::Code::Ok};
}

bool Matches(const Blob &blob, const std::span<const std::byte> bytes,
             const ::rund::StableHash payload_hash) {
  if (blob.uncompressed_bytes != bytes.size() ||
      blob.payload_hash.value != payload_hash.value) {
    return false;
  }
  if (blob.codec == Codec::Raw) {
    const std::span<const std::byte> encoded = blob.encoded.span();
    return encoded.size() == bytes.size() &&
           std::equal(encoded.begin(), encoded.end(), bytes.begin());
  }
  return RleMatches(blob.encoded.span(), bytes);
}

bool VerifiedMatches(const Blob &blob, const std::span<const std::byte> bytes,
                     const ::rund::StableHash payload_hash,
                     ByteHash *const record_hash) {
  if (blob.uncompressed_bytes != bytes.size() ||
      blob.payload_hash.value != payload_hash.value) {
    return false;
  }
  if (blob.codec == Codec::Rle) {
    return RleMatchesAndHash(blob.encoded.span(), bytes, payload_hash.value,
                             record_hash);
  }
  const std::span<const std::byte> encoded = blob.encoded.span();
  if (encoded.size() != bytes.size()) {
    return false;
  }
  if (!std::equal(encoded.begin(), encoded.end(), bytes.begin())) {
    return false;
  }
  ByteHash hash{};
  hash.Append(encoded);
  if (record_hash != nullptr) {
    record_hash->Append(encoded);
  }
  return hash.Finish() == payload_hash.value;
}

} // namespace rund::node::replay_detail::payload
