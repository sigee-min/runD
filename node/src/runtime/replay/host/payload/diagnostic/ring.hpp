#pragma once

#include <node/runtime/replay/host/archive.hpp>
#include <rund/replay/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace rund::node::replay_detail::payload {

struct RawByteSource final {
  const void *context = nullptr;
  std::size_t slice_count = 0u;
  std::uint64_t admitted_bytes = 0u;
  std::uint64_t byte_count = 0u;
  std::span<const std::byte> (*slice)(const void *,
                                      std::size_t) noexcept = nullptr;
};

[[nodiscard]] ::rund::StableHash
HashIngress(const RawByteSource &source) noexcept;

class RawCaptureRing final {
public:
  RawCaptureRing() noexcept = default;
  explicit RawCaptureRing(::rund::replay::Diagnostic config);
  RawCaptureRing(const RawCaptureRing &) = delete;
  RawCaptureRing &operator=(const RawCaptureRing &) = delete;
  RawCaptureRing(RawCaptureRing &&) noexcept = default;
  RawCaptureRing &operator=(RawCaptureRing &&) noexcept = default;

  [[nodiscard]] bool enabled() const noexcept;
  [[nodiscard]] ::rund::StableHash
  Capture(std::uint64_t event_sequence, ::rund::host::EventKind kind,
          const RawByteSource &source) noexcept;

  [[nodiscard]] ::rund::node::replay_detail::payload::DiagnosticArchive
  Archive() const;

  void Clear() noexcept;

private:
  struct Record final {
    std::uint64_t event_sequence = 0u;
    ::rund::host::EventKind kind = ::rund::host::EventKind::None;
    std::uint64_t byte_offset = 0u;
    std::uint64_t byte_count = 0u;
    ::rund::StableHash payload_hash{};
  };

  void EvictOldest() noexcept;
  void Copy(std::uint64_t offset, std::span<std::byte> output) const noexcept;
  [[nodiscard]] ::rund::StableHash Retain(std::uint64_t event_sequence,
                                          ::rund::host::EventKind kind,
                                          const RawByteSource &source) noexcept;

  ::rund::replay::Diagnostic config_{};
  std::vector<std::byte> bytes_{};
  std::vector<Record> records_{};
  std::uint64_t byte_head_ = 0u;
  std::uint64_t byte_size_ = 0u;
  std::uint64_t record_head_ = 0u;
  std::uint64_t record_count_ = 0u;
  std::uint64_t evicted_records_ = 0u;
  std::uint64_t dropped_records_ = 0u;
};

[[nodiscard]] bool ValidDiagnosticArchive(
    const ::rund::node::replay_detail::payload::DiagnosticArchive
        &archive) noexcept;

} // namespace rund::node::replay_detail::payload
