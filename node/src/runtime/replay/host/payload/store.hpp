#pragma once

#include <node/runtime/replay/host/archive.hpp>
#include <rund/replay/code.hpp>
#include <rund/replay/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "blob.hpp"
#include "diagnostic/ring.hpp"
#include "hash.hpp"
#include "index.hpp"

namespace rund::node::replay_detail::payload {

class Backend;
struct EncodedResult;

struct Piece {
  std::uint32_t blob_index = 0u;
};

struct Limits final {
  std::size_t hosts = 0u;
  std::size_t inputs = 0u;
  std::size_t pieces = 0u;
  std::size_t blobs = 0u;
  std::size_t staged = 0u;
  std::uint64_t bytes = 0u;

  [[nodiscard]] static std::optional<Limits>
  runtime(std::uint32_t hosts, std::uint32_t inputs,
          std::uint64_t bytes) noexcept;

  [[nodiscard]] static std::optional<Limits> archive(
      const ::rund::node::replay_detail::payload::Archive &archive) noexcept;
};

struct StoredRecord {
  Record metadata{};
  std::uint64_t record_hash = 0u;
  std::uint32_t piece_offset = 0u;
  std::uint32_t piece_count = 0u;
};

struct InputSourceRange final {
  std::uint64_t event_offset = 0u;
  std::uint64_t event_count = 0u;
  std::uint64_t payload_offset = 0u;
  std::uint64_t payload_count = 0u;
  std::uint64_t hash = 0u;
};

struct ResolveResult {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  std::uint64_t sequence = 0u;
  ::rund::node::replay_detail::payload::Bytes bytes{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

struct MatchResult {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

// Runtime validation views are deliberately smaller than persisted Record
// metadata. They describe one lookup request and never own record identity,
// source ranges, or storage state.
struct Binding {
  std::uint64_t event_sequence = 0u;
  ::rund::host::EventKind kind = ::rund::host::EventKind::None;
  std::uint64_t completed_bytes = 0u;
  ::rund::StableHash payload_hash{};
};

struct InputBinding {
  std::uint64_t source = 0u;
  std::uint64_t schema = 0u;
  std::uint64_t sequence = 0u;
};

class Store {
public:
  Store();
  explicit Store(::rund::replay::Storage storage, Limits limits,
                 ::rund::replay::Diagnostic diagnostic = {});
  ~Store();
  Store(const Store &) = delete;
  Store &operator=(const Store &) = delete;
  Store(Store &&) noexcept;
  Store &operator=(Store &&) noexcept;

  [[nodiscard]] bool Append(std::uint64_t event_sequence,
                            ::rund::host::EventKind kind, Capture payload);

  [[nodiscard]] bool Append(std::uint64_t event_sequence,
                            ::rund::host::EventKind kind,
                            ::rund::node::replay_detail::payload::Bytes bytes,
                            Capture payload);

  [[nodiscard]] bool
  AppendInput(std::uint64_t source, std::uint64_t schema,
              std::uint64_t sequence, InputSourceRange source_range,
              ::rund::node::replay_detail::payload::Bytes bytes,
              Capture payload);

  [[nodiscard]] bool CapturesIngress() const noexcept;
  [[nodiscard]] ::rund::StableHash
  CaptureIngress(std::uint64_t event_sequence, ::rund::host::EventKind kind,
                 const RawByteSource &source) noexcept;

  [[nodiscard]] std::optional<std::uint64_t> SourceRangeHash(
      std::uint64_t event_offset, std::span<const ::rund::host::Event> events,
      std::uint64_t payload_offset, std::uint64_t payload_count) const;

  [[nodiscard]] ResolveResult Resolve(std::size_t record_index) const;
  [[nodiscard]] EncodedResult Encoded(std::size_t chunk_index) const noexcept;

  [[nodiscard]] MatchResult ReadInto(std::size_t record_index,
                                     const Binding &binding,
                                     std::span<std::byte> output) const;

  [[nodiscard]] MatchResult Matches(std::size_t record_index,
                                    const Binding &binding,
                                    std::span<const std::byte> expected) const;

  [[nodiscard]] MatchResult ReadInput(std::size_t input_index,
                                      const InputBinding &binding,
                                      std::span<std::byte> output) const;
  [[nodiscard]] MatchResult CheckInput(std::size_t input_index,
                                       const InputBinding &binding) const;

  [[nodiscard]] bool
  LoadArchive(::rund::node::replay_detail::payload::Archive archive);

  [[nodiscard]] std::uint64_t logical_bytes() const noexcept;
  [[nodiscard]] std::uint64_t retained_bytes() const noexcept;
  [[nodiscard]] std::uint64_t encoded_bytes() const noexcept;
  [[nodiscard]] std::uint64_t segment_count() const noexcept;
  [[nodiscard]] std::uint64_t payload_hash() const;
  [[nodiscard]] ::rund::node::replay_detail::payload::Archive Archive() const;
  [[nodiscard]] const ::rund::replay::Storage &storage() const noexcept;
  [[nodiscard]] const Limits &limits() const noexcept;
  [[nodiscard]] const std::vector<StoredRecord> &records() const noexcept;
  [[nodiscard]] std::span<const Piece>
  pieces(const StoredRecord &record) const noexcept;
  [[nodiscard]] std::size_t host_record_count() const noexcept;
  [[nodiscard]] std::size_t input_record_count() const noexcept;
  [[nodiscard]] std::size_t host_record_index(std::size_t index) const noexcept;
  [[nodiscard]] std::size_t
  input_record_index(std::size_t index) const noexcept;
  [[nodiscard]] const std::vector<Blob> &blobs() const noexcept;

  void Clear() noexcept;

private:
  struct Publication final {
    std::uint64_t retained_bytes = 0u;
    std::uint64_t copied_bytes = 0u;
  };

  [[nodiscard]] static Publication Compact(
      std::vector<::rund::node::replay_detail::payload::ArchiveChunk> &chunks);
  void Commit(StoredRecord record, std::uint64_t logical_bytes);
  [[nodiscard]] bool
  AppendRecord(StoredRecord record, Capture payload,
               ::rund::node::replay_detail::payload::Bytes adopted = {});
  [[nodiscard]] MatchResult Read(std::size_t record_index,
                                 std::span<std::byte> output) const;
  [[nodiscard]] bool ComputeRecordHashes();
  void RebuildRoleIndices();

  ::rund::replay::Storage storage_{};
  Limits limits_{};
  std::vector<StoredRecord> records_{};
  std::vector<std::size_t> host_record_indices_{};
  std::vector<std::size_t> input_record_indices_{};
  std::vector<Piece> pieces_{};
  std::vector<Piece> piece_scratch_{};
  std::vector<Blob> staged_blobs_{};
  Index staged_index_{};
  std::unique_ptr<Backend> backend_{};
  RawCaptureRing diagnostic_{};
  ::rund::node::replay_detail::payload::DiagnosticArchive loaded_diagnostic_{};
  mutable ::rund::node::replay_detail::payload::Bytes publication_{};
  mutable std::size_t publication_chunks_ = 0u;
  std::uint64_t logical_bytes_ = 0u;
  std::uint64_t growths_ = 0u;
  std::uint64_t loaded_payload_hash_ = 0u;
  bool loaded_archive_ = false;
  bool loaded_diagnostic_active_ = false;
};

struct BuildResult {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  Store store{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

[[nodiscard]] BuildResult
Build(::rund::node::replay_detail::payload::Archive archive,
      ::rund::replay::Storage storage,
      ::rund::replay::Diagnostic diagnostic = {});

} // namespace rund::node::replay_detail::payload
