#pragma once

#include <node/runtime/replay/codec/save.hpp>

#include <kernel/core/checked.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>

namespace rund::node::replay_detail::artifact {

enum class Kind : std::uint8_t {
  Record = 1u,
  Checkpoint = 2u,
  HostEvents = 3u,
  Accelerator = 4u,
  Kernel = 5u,
};

inline constexpr std::array<std::byte, 6u> kMagic{
    std::byte{'r'}, std::byte{'u'},  std::byte{'n'},
    std::byte{'D'}, std::byte{0x1a}, std::byte{0x0a},
};
inline constexpr std::uint8_t kSchema = 1u;
inline constexpr std::size_t kHeaderBytes = kMagic.size() + 2u;
inline constexpr std::size_t kWriteSpanBytes = 4096u;

[[nodiscard]] constexpr std::size_t VaruintBytes(std::uint64_t value) noexcept {
  std::size_t bytes = 1u;
  while (value >= 0x80u) {
    value >>= 7u;
    ++bytes;
  }
  return bytes;
}

class Writer final {
public:
  explicit Writer(const Sink sink) noexcept
      : sink_(sink),
        code_(sink.valid() ? ::rund::replay::Code::Ok
                           : ::rund::replay::Code::ArtifactOutputInvalid) {}

  [[nodiscard]] bool ok() const noexcept {
    return code_ == ::rund::replay::Code::Ok;
  }

  [[nodiscard]] bool reject(const ::rund::replay::Code code) noexcept {
    if (ok()) {
      code_ = code == ::rund::replay::Code::Ok
                  ? ::rund::replay::Code::ArtifactWriteFailed
                  : code;
    }
    return false;
  }

  [[nodiscard]] bool require(const bool result,
                             const ::rund::replay::Code code) noexcept {
    return result || reject(code);
  }

  [[nodiscard]] bool header(const Kind kind) noexcept {
    return raw(kMagic) && u8(kSchema) && u8(static_cast<std::uint8_t>(kind));
  }

  [[nodiscard]] bool u8(const std::uint8_t value) noexcept {
    if (!room(1u)) {
      return false;
    }
    buffer_[size_++] = static_cast<std::byte>(value);
    return true;
  }

  [[nodiscard]] bool varuint(std::uint64_t value) noexcept {
    do {
      const std::uint8_t byte = static_cast<std::uint8_t>(value & 0x7fu);
      value >>= 7u;
      if (!u8(static_cast<std::uint8_t>(byte | (value == 0u ? 0u : 0x80u)))) {
        return false;
      }
    } while (value != 0u);
    return true;
  }

  [[nodiscard]] bool sint(const std::int64_t value) noexcept {
    const std::uint64_t bits = std::bit_cast<std::uint64_t>(value);
    return varuint((bits << 1u) ^ (0u - (bits >> 63u)));
  }

  [[nodiscard]] bool fixed64(const std::uint64_t value) noexcept {
    if (!room(sizeof(value))) {
      return false;
    }
    for (unsigned shift = 0u; shift < 64u; shift += 8u) {
      buffer_[size_++] =
          static_cast<std::byte>((value >> shift) & std::uint64_t{0xffu});
    }
    return true;
  }

  // Payload and checkpoint state bypass the staging buffer. The caller-owned
  // bytes are exposed to the sink in bounded subspans, so saving performs no
  // internal payload-sized copy.
  [[nodiscard]] bool raw(std::span<const std::byte> bytes) noexcept {
    if (!flush()) {
      return false;
    }
    while (ok() && !bytes.empty()) {
      const std::size_t count = std::min(bytes.size(), kWriteSpanBytes);
      if (!emit(bytes.first(count))) {
        return false;
      }
      bytes = bytes.subspan(count);
    }
    return ok();
  }

  [[nodiscard]] Result finish() noexcept {
    if (ok()) {
      static_cast<void>(flush());
    }
    return Result{.code = code_, .bytes = bytes_, .writes = writes_};
  }

private:
  [[nodiscard]] bool room(const std::size_t bytes) noexcept {
    if (!ok()) {
      return false;
    }
    if (bytes > buffer_.size()) {
      return reject(::rund::replay::Code::ArtifactCapacityExceeded);
    }
    return buffer_.size() - size_ >= bytes || flush();
  }

  [[nodiscard]] bool emit(const std::span<const std::byte> bytes) noexcept {
    if (!ok() || bytes.empty()) {
      return ok();
    }
    if constexpr (sizeof(std::size_t) > sizeof(std::uint64_t)) {
      if (bytes.size() >
          static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
        return reject(::rund::replay::Code::ArtifactCapacityExceeded);
      }
    }
    std::uint64_t next_bytes = 0u;
    std::uint64_t next_writes = 0u;
    if (!rund::kernel::checked::add(
            bytes_, static_cast<std::uint64_t>(bytes.size()), next_bytes) ||
        next_bytes > sink_.max_bytes ||
        !rund::kernel::checked::add(writes_, 1u, next_writes)) {
      return reject(::rund::replay::Code::ArtifactCapacityExceeded);
    }
    if (!sink_.write(sink_.state, bytes)) {
      return reject(::rund::replay::Code::ArtifactWriteFailed);
    }
    bytes_ = next_bytes;
    writes_ = next_writes;
    return true;
  }

  [[nodiscard]] bool flush() noexcept {
    if (size_ == 0u) {
      return ok();
    }
    if (!emit(std::span<const std::byte>{buffer_.data(), size_})) {
      return false;
    }
    size_ = 0u;
    return true;
  }

  Sink sink_{};
  ::rund::replay::Code code_ = ::rund::replay::Code::ArtifactNotSaved;
  std::array<std::byte, kWriteSpanBytes> buffer_{};
  std::size_t size_ = 0u;
  std::uint64_t bytes_ = 0u;
  std::uint64_t writes_ = 0u;
};

class Reader final {
public:
  explicit constexpr Reader(const std::span<const std::byte> bytes) noexcept
      : bytes_(bytes) {}

  [[nodiscard]] bool header(const Kind expected) noexcept {
    std::span<const std::byte> magic{};
    std::uint8_t schema = 0u;
    std::uint8_t kind = 0u;
    return take(kMagic.size(), magic) &&
           std::equal(magic.begin(), magic.end(), kMagic.begin()) &&
           u8(schema) && schema == kSchema && u8(kind) &&
           kind == static_cast<std::uint8_t>(expected);
  }

  [[nodiscard]] bool u8(std::uint8_t &value) noexcept {
    if (cursor_ == bytes_.size()) {
      return false;
    }
    value = std::to_integer<std::uint8_t>(bytes_[cursor_++]);
    return true;
  }

  [[nodiscard]] bool varuint(std::uint64_t &value) noexcept {
    value = 0u;
    for (std::size_t index = 0u; index < 10u; ++index) {
      std::uint8_t byte = 0u;
      if (!u8(byte) || (index == 9u && byte > 1u)) {
        return false;
      }
      value |= static_cast<std::uint64_t>(byte & 0x7fu) << (index * 7u);
      if ((byte & 0x80u) == 0u) {
        return VaruintBytes(value) == index + 1u;
      }
    }
    return false;
  }

  [[nodiscard]] bool sint(std::int64_t &value) noexcept {
    std::uint64_t encoded = 0u;
    if (!varuint(encoded)) {
      return false;
    }
    const std::uint64_t bits =
        (encoded >> 1u) ^ (0u - (encoded & std::uint64_t{1u}));
    value = std::bit_cast<std::int64_t>(bits);
    return true;
  }

  [[nodiscard]] bool fixed64(std::uint64_t &value) noexcept {
    if (bytes_.size() - cursor_ < sizeof(value)) {
      return false;
    }
    value = 0u;
    for (unsigned shift = 0u; shift < 64u; shift += 8u) {
      value |= static_cast<std::uint64_t>(
                   std::to_integer<std::uint8_t>(bytes_[cursor_++]))
               << shift;
    }
    return true;
  }

  [[nodiscard]] bool take(const std::size_t count,
                          std::span<const std::byte> &out) noexcept {
    if (count > bytes_.size() - cursor_) {
      return false;
    }
    out = bytes_.subspan(cursor_, count);
    cursor_ += count;
    return true;
  }

  [[nodiscard]] constexpr std::size_t position() const noexcept {
    return cursor_;
  }

  [[nodiscard]] bool view(const std::size_t offset, const std::size_t count,
                          std::span<const std::byte> &out) const noexcept {
    if (offset > bytes_.size() || count > bytes_.size() - offset) {
      return false;
    }
    out = bytes_.subspan(offset, count);
    return true;
  }

  [[nodiscard]] constexpr bool done() const noexcept {
    return cursor_ == bytes_.size();
  }

  [[nodiscard]] constexpr std::size_t remaining() const noexcept {
    return bytes_.size() - cursor_;
  }

private:
  std::span<const std::byte> bytes_{};
  std::size_t cursor_ = 0u;
};

} // namespace rund::node::replay_detail::artifact
