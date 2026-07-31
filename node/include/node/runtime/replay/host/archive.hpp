#pragma once

#include <node/runtime/replay/host/bytes.hpp>
#include <node/runtime/replay/host/record.hpp>
#include <rund/replay/storage.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace rund::node::replay_detail::payload {

class SpillGeneration;

enum class Codec : std::uint8_t {
  Raw = 0u,
  Rle = 1u,
};

enum class DiagnosticRole : std::uint8_t {
  NetworkIngress = 0u,
};

struct ArchivePiece {
  std::uint64_t chunk_id = 0u;
  std::uint32_t offset = 0u;
  std::uint32_t size = 0u;
};

struct ArchiveChunk {
  std::uint64_t chunk_id = 0u;
  Codec codec = Codec::Raw;
  std::uint64_t uncompressed_bytes = 0u;
  std::uint64_t encoded_bytes = 0u;
  ::rund::StableHash uncompressed_hash{};
  Bytes encoded{};
  bool spilled = false;
  std::uint32_t segment_index = 0u;
  std::uint64_t segment_offset = 0u;
  std::uint64_t segment_record_bytes = 0u;
};

struct ArchiveRecord {
  Record metadata{};
  std::vector<ArchivePiece> pieces{};
};

struct DiagnosticRecord {
  DiagnosticRole role = DiagnosticRole::NetworkIngress;
  std::uint64_t event_sequence = 0u;
  ::rund::host::EventKind kind = ::rund::host::EventKind::None;
  std::uint64_t offset = 0u;
  std::uint64_t byte_count = 0u;
  ::rund::StableHash payload_hash{};
};

struct DiagnosticArchive {
  std::vector<DiagnosticRecord> records{};
  Bytes bytes{};
  std::uint64_t hash = 0u;
  ::rund::replay::DiagnosticReport report{};
};

struct Archive {
  std::vector<ArchiveRecord> records{};
  std::vector<ArchiveChunk> chunks{};
  std::uint64_t payload_hash = 0u;
  ::rund::replay::StorageReport storage{};
  DiagnosticArchive diagnostic{};
  // Spill coordinates are meaningful only while this immutable generation
  // owner lives. The owner is process-local and is never serialized or hashed.
  std::shared_ptr<const SpillGeneration> spill_generation{};
};

} // namespace rund::node::replay_detail::payload
