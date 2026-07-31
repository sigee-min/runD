#pragma once

#include "blob.hpp"
#include "cache.hpp"
#include "chunk.hpp"
#include "codec.hpp"
#include "index.hpp"

#include <node/runtime/replay/host/bytes.hpp>
#include <rund/host/hash.hpp>
#include <rund/replay/code.hpp>
#include <rund/replay/storage.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rund::node::replay_detail::payload {

class ByteHash;

class SpillGeneration final {
public:
  [[nodiscard]] static std::shared_ptr<const SpillGeneration>
  Create(const std::string &root, ::rund::storage::Budget budget,
         std::size_t reservation_capacity) noexcept;

  ~SpillGeneration();
  SpillGeneration(const SpillGeneration &) = delete;
  SpillGeneration &operator=(const SpillGeneration &) = delete;
  SpillGeneration(SpillGeneration &&) = delete;
  SpillGeneration &operator=(SpillGeneration &&) = delete;

  [[nodiscard]] const std::string &directory() const noexcept {
    return directory_;
  }
  [[nodiscard]] ::rund::storage::Reservation
  Reserve(std::uint64_t allocated_bytes) const noexcept;
  [[nodiscard]] bool
  Stage(::rund::storage::Reservation reservation) const noexcept;
  [[nodiscard]] bool CommitLast(::rund::storage::Usage usage) const noexcept;
  void Rollback(std::size_t reservation_count) const noexcept;
  [[nodiscard]] std::size_t reservation_count() const noexcept;
  [[nodiscard]] ::rund::storage::Usage usage() const noexcept;
  [[nodiscard]] std::uint64_t reserved_bytes() const noexcept;

private:
  SpillGeneration(
      std::string root, std::string directory, int lease_descriptor,
      ::rund::storage::Budget budget,
      std::vector<::rund::storage::Reservation> reservations) noexcept
      : root_(std::move(root)), directory_(std::move(directory)),
        lease_descriptor_(lease_descriptor), budget_(std::move(budget)),
        reservations_(std::move(reservations)) {}

  std::string root_{};
  std::string directory_{};
  int lease_descriptor_ = -1;
  ::rund::storage::Budget budget_{};
  mutable std::vector<::rund::storage::Reservation> reservations_{};
  mutable ::rund::storage::Usage usage_{};
  mutable std::uint64_t reserved_bytes_ = 0u;
};

struct ReadResult {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  std::vector<std::byte> bytes{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

struct EncodedResult final {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  ::rund::node::replay_detail::payload::Bytes bytes{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

struct ReadStatus {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

struct AppendResult {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  std::uint32_t blob_index = 0u;
  std::uint64_t encoded_bytes = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

struct BatchResult {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  std::uint32_t first_blob = 0u;
  std::uint32_t blob_count = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

struct MemoryMark {
  std::size_t blobs = 0u;
  std::uint64_t retained_bytes = 0u;
  std::uint64_t encoded_bytes = 0u;
};

[[nodiscard]] Blob Encode(std::span<const std::byte> bytes,
                          ::rund::StableHash payload_hash);

[[nodiscard]] ReadResult Decode(::rund::StableHash payload_hash,
                                std::uint64_t uncompressed_bytes, Codec codec,
                                std::span<const std::byte> encoded);

[[nodiscard]] ReadStatus DecodeInto(::rund::StableHash payload_hash,
                                    std::uint64_t uncompressed_bytes,
                                    Codec codec,
                                    std::span<const std::byte> encoded,
                                    std::span<std::byte> output,
                                    ByteHash *record_hash = nullptr) noexcept;

[[nodiscard]] bool Matches(const Blob &blob, std::span<const std::byte> bytes,
                           ::rund::StableHash payload_hash);

[[nodiscard]] bool VerifiedMatches(const Blob &blob,
                                   std::span<const std::byte> bytes,
                                   ::rund::StableHash payload_hash,
                                   ByteHash *record_hash = nullptr);

class Memory {
public:
  Memory() = default;
  explicit Memory(std::size_t capacity);
  Memory(const Memory &) = delete;
  Memory &operator=(const Memory &) = delete;
  Memory(Memory &&) noexcept = default;
  Memory &operator=(Memory &&) noexcept = default;

  [[nodiscard]] std::uint32_t Append(Blob blob);
  [[nodiscard]] bool CanAppend(const Blob &blob) const noexcept;

  [[nodiscard]] ReadResult Read(std::uint32_t blob_index) const;
  [[nodiscard]] EncodedResult Encoded(std::uint32_t blob_index) const noexcept;
  [[nodiscard]] ReadStatus
  ReadInto(std::uint32_t blob_index, std::span<std::byte> output,
           ByteHash *record_hash = nullptr) const noexcept;

  [[nodiscard]] std::uint64_t retained_bytes() const noexcept;
  [[nodiscard]] std::uint64_t encoded_bytes() const noexcept;
  [[nodiscard]] std::uint64_t growths() const noexcept;
  [[nodiscard]] const std::vector<Blob> &blobs() const noexcept;
  [[nodiscard]] MemoryMark Mark() const noexcept;
  void Rollback(MemoryMark mark) noexcept;

  void Clear() noexcept;

private:
  std::vector<Blob> blobs_{};
  std::uint64_t retained_bytes_ = 0u;
  std::uint64_t encoded_bytes_ = 0u;
  std::uint64_t growths_ = 0u;
};

struct SpillRef {
  std::uint32_t segment_index = 0u;
  std::uint64_t segment_offset = 0u;
  std::uint64_t record_bytes = 0u;
};

struct SpillMark {
  std::size_t blobs = 0u;
  std::size_t refs = 0u;
  std::size_t reservations = 0u;
  std::uint64_t encoded_bytes = 0u;
  std::uint32_t segment = 0u;
  std::uint64_t segment_bytes = 0u;
};

class Spill {
public:
  Spill();
  Spill(const ::rund::replay::Storage &storage, std::size_t capacity);
  Spill(const Spill &) = delete;
  Spill &operator=(const Spill &) = delete;
  Spill(Spill &&) noexcept = default;
  Spill &operator=(Spill &&) noexcept = default;

  [[nodiscard]] AppendResult Append(Blob blob);
  [[nodiscard]] bool CanAppend(const Blob &blob) const noexcept;
  [[nodiscard]] SpillMark Mark() const noexcept;
  [[nodiscard]] bool Rollback(SpillMark mark) noexcept;

  [[nodiscard]] ReadResult Read(std::uint32_t blob_index) const;
  [[nodiscard]] EncodedResult Encoded(std::uint32_t blob_index) const noexcept;
  [[nodiscard]] ReadStatus ReadInto(std::uint32_t blob_index,
                                    std::span<std::byte> output,
                                    ByteHash *record_hash = nullptr) const;
  [[nodiscard]] ReadStatus Matches(std::uint32_t blob_index,
                                   std::span<const std::byte> expected,
                                   ByteHash *record_hash = nullptr) const;

  [[nodiscard]] std::uint64_t retained_bytes() const noexcept;
  [[nodiscard]] std::uint64_t encoded_bytes() const noexcept;
  [[nodiscard]] const std::vector<Blob> &blobs() const noexcept;
  [[nodiscard]] std::uint64_t segment_count() const noexcept;
  [[nodiscard]] const std::vector<SpillRef> &refs() const noexcept;
  [[nodiscard]] ::rund::replay::StorageReport Report() const noexcept;
  [[nodiscard]] const std::shared_ptr<const SpillGeneration> &
  generation() const noexcept;

  void LoadArchive(std::vector<Blob> blobs, std::vector<SpillRef> refs,
                   std::uint64_t encoded_bytes,
                   std::shared_ptr<const SpillGeneration> generation) noexcept;
  void Clear() noexcept;

private:
  [[nodiscard]] bool EnsureGeneration() noexcept;

  ::rund::replay::Storage storage_{};
  std::vector<Blob> blobs_{};
  std::vector<SpillRef> refs_{};
  std::shared_ptr<const SpillGeneration> generation_{};
  mutable Cache cache_{0u};
  std::uint64_t encoded_bytes_ = 0u;
  std::uint64_t growths_ = 0u;
  std::uint32_t current_segment_index_ = 0u;
  std::uint64_t current_segment_bytes_ = 0u;
};

class Backend {
public:
  Backend();
  Backend(const ::rund::replay::Storage &storage, std::size_t capacity);
  Backend(const Backend &) = delete;
  Backend &operator=(const Backend &) = delete;
  Backend(Backend &&) noexcept = default;
  Backend &operator=(Backend &&) noexcept = default;

  [[nodiscard]] std::optional<std::uint32_t>
  Find(std::span<const std::byte> bytes, ::rund::StableHash payload_hash) const;

  [[nodiscard]] BatchResult Append(std::vector<Blob> &blobs);
  [[nodiscard]] bool CanAppend(std::span<const Blob> blobs) const noexcept;

  [[nodiscard]] ReadResult Read(std::uint32_t blob_index) const;
  [[nodiscard]] EncodedResult Encoded(std::uint32_t blob_index) const noexcept;
  [[nodiscard]] ReadStatus ReadInto(std::uint32_t blob_index,
                                    std::span<std::byte> output,
                                    ByteHash *record_hash = nullptr) const;

  [[nodiscard]] ReadStatus Matches(std::uint32_t blob_index,
                                   std::span<const std::byte> expected,
                                   ByteHash *record_hash = nullptr) const;

  [[nodiscard]] std::uint64_t retained_bytes() const noexcept;
  [[nodiscard]] std::uint64_t encoded_bytes() const noexcept;
  [[nodiscard]] const std::vector<Blob> &blobs() const noexcept;
  [[nodiscard]] std::uint64_t segment_count() const noexcept;
  [[nodiscard]] const std::vector<SpillRef> &refs() const noexcept;
  [[nodiscard]] ::rund::replay::StorageReport Report() const noexcept;
  [[nodiscard]] const std::shared_ptr<const SpillGeneration> &
  generation() const noexcept;

  void LoadArchive(std::vector<Blob> blobs, std::vector<SpillRef> refs,
                   std::uint64_t encoded_bytes,
                   std::shared_ptr<const SpillGeneration> generation) noexcept;
  void Clear() noexcept;

private:
  [[nodiscard]] bool index(std::uint32_t blob_index,
                           ::rund::StableHash payload_hash) noexcept;
  [[nodiscard]] bool rebuild() noexcept;

  ::rund::replay::Storage storage_{};
  Memory memory_{};
  Spill spill_;
  payload::Index index_{};
  bool writable_ = true;
};

} // namespace rund::node::replay_detail::payload
