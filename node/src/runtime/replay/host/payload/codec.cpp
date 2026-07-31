#include "codec.hpp"
#include "hash.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace rund::node::replay_detail::payload {
namespace {

[[nodiscard]] std::size_t
RepeatRunLength(const std::span<const std::byte> bytes,
                const std::size_t offset) noexcept {
  std::size_t length = 1u;
  while (offset + length < bytes.size() && length < 130u &&
         bytes[offset + length] == bytes[offset]) {
    ++length;
  }
  return length;
}

template <typename Repeat, typename Literal>
[[nodiscard]] bool WalkRle(const std::span<const std::byte> encoded,
                           const std::size_t output_size, Repeat repeat,
                           Literal literal) noexcept {
  std::size_t index = 0u;
  std::size_t written = 0u;
  while (index < encoded.size()) {
    const std::uint8_t tag = std::to_integer<std::uint8_t>(encoded[index++]);
    if ((tag & 0x80u) != 0u) {
      const std::size_t length = (tag & 0x7fu) + 3u;
      if (index >= encoded.size() || length > output_size - written ||
          !repeat(written, length, encoded[index++])) {
        return false;
      }
      written += length;
      continue;
    }
    const std::size_t length = tag + 1u;
    if (length > encoded.size() - index || length > output_size - written ||
        !literal(written, encoded.subspan(index, length))) {
      return false;
    }
    index += length;
    written += length;
  }
  return written == output_size;
}

template <typename Append>
[[nodiscard]] bool
WalkDecoded(const std::uint64_t uncompressed_bytes, const Codec codec,
            const std::span<const std::byte> encoded, Append append) noexcept {
  if (uncompressed_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return false;
  }
  const std::size_t output_size =
      static_cast<std::size_t>(uncompressed_bytes);
  switch (codec) {
  case Codec::Raw:
    if (encoded.size() != output_size) {
      return false;
    }
    append(encoded);
    return true;
  case Codec::Rle: {
    std::array<std::byte, 130u> repeated{};
    return WalkRle(
        encoded, output_size,
        [&append, &repeated](const std::size_t, const std::size_t length,
                             const std::byte value) noexcept {
          std::fill_n(repeated.begin(), length, value);
          append(std::span<const std::byte>{repeated}.first(length));
          return true;
        },
        [&append](const std::size_t,
                  const std::span<const std::byte> literal) noexcept {
          append(literal);
          return true;
        });
  }
  }
  return false;
}

} // namespace

std::vector<std::byte> EncodeRle(const std::span<const std::byte> bytes) {
  std::vector<std::byte> encoded{};
  std::size_t index = 0u;
  while (index < bytes.size()) {
    const std::size_t repeat = RepeatRunLength(bytes, index);
    if (repeat >= 3u) {
      encoded.push_back(static_cast<std::byte>(0x80u | (repeat - 3u)));
      encoded.push_back(bytes[index]);
      index += repeat;
      continue;
    }
    const std::size_t literal_begin = index;
    std::size_t literal_length = 0u;
    while (index < bytes.size() && literal_length < 128u) {
      if (RepeatRunLength(bytes, index) >= 3u) {
        break;
      }
      ++index;
      ++literal_length;
    }
    encoded.push_back(static_cast<std::byte>(literal_length - 1u));
    encoded.insert(encoded.end(), bytes.begin() + literal_begin,
                   bytes.begin() + literal_begin + literal_length);
  }
  return encoded;
}

std::optional<std::vector<std::byte>>
DecodeRle(const std::span<const std::byte> encoded,
          const std::uint64_t expected_uncompressed_bytes) {
  if (expected_uncompressed_bytes >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  std::vector<std::byte> bytes(
      static_cast<std::size_t>(expected_uncompressed_bytes));
  if (!DecodeRleInto(encoded, bytes)) {
    return std::nullopt;
  }
  return bytes;
}

bool DecodeRleInto(const std::span<const std::byte> encoded,
                   const std::span<std::byte> output,
                   ByteHash *const chunk_hash,
                   ByteHash *const record_hash) noexcept {
  return WalkRle(
      encoded, output.size(),
      [output, chunk_hash, record_hash](const std::size_t offset,
                                        const std::size_t length,
                                        const std::byte value) noexcept {
        const std::span<std::byte> written = output.subspan(offset, length);
        std::fill(written.begin(), written.end(), value);
        if (chunk_hash != nullptr) {
          chunk_hash->Append(written);
        }
        if (record_hash != nullptr) {
          record_hash->Append(written);
        }
        return true;
      },
      [output, chunk_hash,
       record_hash](const std::size_t offset,
                    const std::span<const std::byte> literal) noexcept {
        std::copy(literal.begin(), literal.end(), output.begin() + offset);
        if (chunk_hash != nullptr) {
          chunk_hash->Append(literal);
        }
        if (record_hash != nullptr) {
          record_hash->Append(literal);
        }
        return true;
      });
}

bool RleMatches(const std::span<const std::byte> encoded,
                const std::span<const std::byte> expected) noexcept {
  return WalkRle(
      encoded, expected.size(),
      [expected](const std::size_t offset, const std::size_t length,
                 const std::byte value) noexcept {
        const auto begin =
            expected.begin() + static_cast<std::ptrdiff_t>(offset);
        return std::all_of(begin, begin + static_cast<std::ptrdiff_t>(length),
                           [value](const std::byte candidate) noexcept {
                             return candidate == value;
                           });
      },
      [expected](const std::size_t offset,
                 const std::span<const std::byte> literal) noexcept {
        return std::equal(literal.begin(), literal.end(),
                          expected.begin() +
                              static_cast<std::ptrdiff_t>(offset));
      });
}

bool RleMatchesAndHash(const std::span<const std::byte> encoded,
                       const std::span<const std::byte> expected,
                       const std::uint64_t expected_hash,
                       ByteHash *const record_hash) noexcept {
  ByteHash hash{};
  const bool matches = WalkRle(
      encoded, expected.size(),
      [expected, &hash, record_hash](const std::size_t offset,
                                     const std::size_t length,
                                     const std::byte value) noexcept {
        const std::span<const std::byte> matched =
            expected.subspan(offset, length);
        if (!std::all_of(matched.begin(), matched.end(),
                         [value](const std::byte candidate) noexcept {
                           return candidate == value;
                         })) {
          return false;
        }
        hash.Append(matched);
        if (record_hash != nullptr) {
          record_hash->Append(matched);
        }
        return true;
      },
      [expected, &hash,
       record_hash](const std::size_t offset,
                    const std::span<const std::byte> literal) noexcept {
        const std::span<const std::byte> matched =
            expected.subspan(offset, literal.size());
        if (!std::equal(literal.begin(), literal.end(), matched.begin())) {
          return false;
        }
        hash.Append(literal);
        if (record_hash != nullptr) {
          record_hash->Append(literal);
        }
        return true;
      });
  return matches && hash.Finish() == expected_hash;
}

bool Verify(const ::rund::StableHash expected_hash,
            const std::uint64_t uncompressed_bytes, const Codec codec,
            const std::span<const std::byte> encoded,
            ByteHash *const payload_hash) noexcept {
  ByteHash chunk_hash{};
  const auto append = [&chunk_hash, payload_hash](
                          const std::span<const std::byte> bytes) noexcept {
    chunk_hash.Append(bytes);
    if (payload_hash != nullptr) {
      payload_hash->Append(bytes);
    }
  };
  return WalkDecoded(uncompressed_bytes, codec, encoded, append) &&
         chunk_hash.Finish() == expected_hash.value;
}

bool AppendDecodedBytes(const std::uint64_t uncompressed_bytes,
                        const Codec codec,
                        const std::span<const std::byte> encoded,
                        ByteHash &payload_hash) noexcept {
  return WalkDecoded(
      uncompressed_bytes, codec, encoded,
      [&payload_hash](const std::span<const std::byte> bytes) noexcept {
        payload_hash.Append(bytes);
      });
}

} // namespace rund::node::replay_detail::payload
