#include "local.hpp"

#include <array>
#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <optional>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/uio.h>
#include <unistd.h>

namespace rund::node::replay_detail::payload::spill {
namespace {

[[nodiscard]] std::optional<std::uint64_t>
round_up(const std::uint64_t value, const std::uint64_t unit) noexcept {
  if (unit == 0u) {
    return std::nullopt;
  }
  const std::uint64_t remainder = value % unit;
  if (remainder == 0u) {
    return value;
  }
  const std::uint64_t added = unit - remainder;
  std::uint64_t rounded = 0u;
  return rund::kernel::checked::add(value, added, rounded)
             ? std::optional<std::uint64_t>{rounded}
             : std::nullopt;
}

[[nodiscard]] std::uint64_t available_bytes(const std::uint64_t blocks,
                                            const std::uint64_t unit) noexcept {
  std::uint64_t bytes = 0u;
  return rund::kernel::checked::mul(blocks, unit, bytes)
             ? bytes
             : std::numeric_limits<std::uint64_t>::max();
}

void encode_u64(std::byte *const output, const std::uint64_t value) noexcept {
  for (std::uint32_t byte = 0u; byte < sizeof(value); ++byte) {
    output[byte] = static_cast<std::byte>((value >> (byte * 8u)) & 0xffu);
  }
}

[[nodiscard]] std::uint64_t decode_u64(const std::byte *const input) noexcept {
  std::uint64_t value = 0u;
  for (std::uint32_t byte = 0u; byte < sizeof(value); ++byte) {
    value |=
        static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(input[byte]))
        << (byte * 8u);
  }
  return value;
}

[[nodiscard]] bool close_file(const int descriptor) noexcept {
  return descriptor >= 0 && ::close(descriptor) == 0;
}

[[nodiscard]] bool read_all(const int descriptor, std::byte *output,
                            std::size_t bytes, std::uint64_t offset) noexcept {
  while (bytes != 0u) {
    if (offset >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return false;
    }
    const std::size_t request =
        std::min(bytes, static_cast<std::size_t>(
                            std::numeric_limits<ssize_t>::max()));
    const ssize_t read =
        ::pread(descriptor, output, request, static_cast<off_t>(offset));
    if (read > 0) {
      const auto completed = static_cast<std::size_t>(read);
      output += completed;
      bytes -= completed;
      offset += completed;
      continue;
    }
    if (read < 0 && errno == EINTR) {
      continue;
    }
    return false;
  }
  return true;
}

[[nodiscard]] bool write_all(const int descriptor,
                             std::span<const std::byte> header,
                             std::span<const std::byte> payload,
                             std::uint64_t offset) noexcept {
  std::array<iovec, 2u> parts{{
      {.iov_base = const_cast<std::byte *>(header.data()),
       .iov_len = header.size()},
      {.iov_base = const_cast<std::byte *>(payload.data()),
       .iov_len = payload.size()},
  }};
  std::size_t first = 0u;
  while (first < parts.size()) {
    while (first < parts.size() && parts[first].iov_len == 0u) {
      ++first;
    }
    if (first == parts.size()) {
      return true;
    }
    if (offset >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return false;
    }

    const ssize_t written = ::pwritev(descriptor, parts.data() + first,
                                      static_cast<int>(parts.size() - first),
                                      static_cast<off_t>(offset));
    if (written < 0 && errno == EINTR) {
      continue;
    }
    if (written <= 0) {
      return false;
    }

    auto remaining = static_cast<std::size_t>(written);
    offset += remaining;
    while (first < parts.size() && remaining >= parts[first].iov_len) {
      remaining -= parts[first].iov_len;
      ++first;
    }
    if (first < parts.size() && remaining != 0u) {
      auto *const bytes = static_cast<std::byte *>(parts[first].iov_base);
      parts[first].iov_base = bytes + remaining;
      parts[first].iov_len -= remaining;
    }
  }
  return true;
}

[[nodiscard]] std::array<std::byte, kHeaderBytes>
header(const std::uint32_t blob_index, const Blob &blob) noexcept {
  std::array<std::byte, kHeaderBytes> bytes{};
  encode_u64(bytes.data(), kRecordMagic);
  encode_u64(bytes.data() + 8u, blob_index);
  bytes[16u] = static_cast<std::byte>(blob.codec);
  encode_u64(bytes.data() + 17u, blob.uncompressed_bytes);
  encode_u64(bytes.data() + 25u, blob.encoded_bytes);
  encode_u64(bytes.data() + 33u, blob.payload_hash.value);
  return bytes;
}

} // namespace

std::optional<Space> inspect(const std::string &directory,
                             const std::uint64_t segment_bytes,
                             const std::uint64_t record_bytes,
                             const std::uint64_t minimum_free_bytes) noexcept {
  if (!rund::kernel::checked::add(segment_bytes, record_bytes)) {
    return std::nullopt;
  }

  struct statvfs status{};
  if (::statvfs(directory.c_str(), &status) != 0) {
    return std::nullopt;
  }
  const std::uint64_t unit = status.f_frsize != 0u
                                 ? static_cast<std::uint64_t>(status.f_frsize)
                                 : static_cast<std::uint64_t>(status.f_bsize);
  const std::optional<std::uint64_t> before = round_up(segment_bytes, unit);
  const std::optional<std::uint64_t> after =
      round_up(segment_bytes + record_bytes, unit);
  if (!before.has_value() || !after.has_value() || *after < *before) {
    return std::nullopt;
  }

  const std::uint64_t allocation = *after - *before;
  const std::uint64_t available =
      available_bytes(static_cast<std::uint64_t>(status.f_bavail), unit);
  const bool headroom =
      rund::kernel::checked::add(allocation, minimum_free_bytes) &&
      available >= allocation + minimum_free_bytes;
  return Space{.allocation = allocation, .headroom = headroom};
}

std::string path(const std::shared_ptr<const SpillGeneration> &generation,
                 const std::uint32_t segment_index) {
  if (generation == nullptr) {
    return {};
  }
  std::array<char, 64u> name{};
  const int length = std::snprintf(name.data(), name.size(),
                                   "host-replay-payload-%06u.segment",
                                   static_cast<unsigned int>(segment_index));
  if (length < 0 || static_cast<std::size_t>(length) >= name.size()) {
    return {};
  }
  return (std::filesystem::path{generation->directory()} / name.data())
      .string();
}

bool write(const std::shared_ptr<const SpillGeneration> &generation,
           const std::uint32_t segment_index,
           const std::uint64_t segment_offset, const std::uint32_t blob_index,
           const Blob &blob) noexcept {
  try {
    const std::span<const std::byte> encoded = blob.encoded.span();
    if (encoded.size() != blob.encoded_bytes ||
        segment_offset >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      return false;
    }
    const std::string segment_path = path(generation, segment_index);
    if (segment_path.empty()) {
      return false;
    }
    const int flags =
        O_WRONLY | O_CREAT | O_CLOEXEC | (segment_offset == 0u ? O_TRUNC : 0);
    const int descriptor =
        ::open(segment_path.c_str(), flags, static_cast<mode_t>(0666));
    if (descriptor < 0) {
      return false;
    }

    struct stat status{};
    const bool positioned =
        ::fstat(descriptor, &status) == 0 && status.st_size >= 0 &&
        static_cast<std::uint64_t>(status.st_size) == segment_offset;
    const std::array<std::byte, kHeaderBytes> record_header =
        header(blob_index, blob);
    const bool written = positioned && write_all(descriptor, record_header,
                                                 encoded, segment_offset);
    return close_file(descriptor) && written;
  } catch (...) {
    return false;
  }
}

Segment read(const std::shared_ptr<const SpillGeneration> &generation,
             const std::span<const SpillRef> refs,
             const std::span<const Blob> blobs,
             const std::uint32_t blob_index) {
  if (generation == nullptr || blob_index >= refs.size() ||
      blob_index >= blobs.size()) {
    return Segment{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  const SpillRef ref = refs[blob_index];
  if (ref.record_bytes < kHeaderBytes ||
      !rund::kernel::checked::add(ref.segment_offset, ref.record_bytes)) {
    return Segment{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  const std::string segment_path = path(generation, ref.segment_index);
  const int descriptor = ::open(segment_path.c_str(), O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) {
    return Segment{.code = ::rund::replay::Code::HostPayloadMissing};
  }

  std::array<std::byte, kHeaderBytes> record_header{};
  const bool header_read = read_all(descriptor, record_header.data(),
                                    record_header.size(), ref.segment_offset);
  Blob blob{};
  const std::uint64_t magic = decode_u64(record_header.data());
  const std::uint64_t chunk_id = decode_u64(record_header.data() + 8u);
  const std::uint8_t codec = std::to_integer<std::uint8_t>(record_header[16u]);
  blob.uncompressed_bytes = decode_u64(record_header.data() + 17u);
  blob.encoded_bytes = decode_u64(record_header.data() + 25u);
  blob.payload_hash.value = decode_u64(record_header.data() + 33u);
  const bool valid_header =
      header_read && magic == kRecordMagic && chunk_id == blob_index &&
      blob.encoded_bytes == ref.record_bytes - kHeaderBytes &&
      blob.encoded_bytes <=
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max());
  if (!valid_header) {
    static_cast<void>(close_file(descriptor));
    return Segment{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  if (codec == static_cast<std::uint8_t>(Codec::Raw)) {
    blob.codec = Codec::Raw;
  } else if (codec == static_cast<std::uint8_t>(Codec::Rle)) {
    blob.codec = Codec::Rle;
  } else {
    static_cast<void>(close_file(descriptor));
    return Segment{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }

  std::byte *encoded_data = nullptr;
  Bytes encoded =
      Bytes::create(static_cast<std::size_t>(blob.encoded_bytes), encoded_data);
  const bool payload_read = read_all(descriptor, encoded_data, encoded.size(),
                                     ref.segment_offset + kHeaderBytes);
  const bool closed = close_file(descriptor);
  if (!payload_read || !closed) {
    return Segment{.code = ::rund::replay::Code::HostPayloadMissing};
  }
  blob.encoded = std::move(encoded);

  const Blob &expected = blobs[blob_index];
  if (blob.uncompressed_bytes != expected.uncompressed_bytes ||
      blob.encoded_bytes != expected.encoded_bytes ||
      blob.payload_hash.value != expected.payload_hash.value ||
      blob.codec != expected.codec) {
    return Segment{.code = ::rund::replay::Code::HostPayloadHashInvalid};
  }
  return Segment{.code = ::rund::replay::Code::Ok, .blob = std::move(blob)};
}

} // namespace rund::node::replay_detail::payload::spill
