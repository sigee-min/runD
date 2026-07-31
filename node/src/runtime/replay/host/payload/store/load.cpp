#include "load/local.hpp"

#include <node/runtime/replay/host/payload.hpp>

#include <utility>

namespace rund::node::replay_detail::payload {

bool Store::LoadArchive(::rund::node::replay_detail::payload::Archive archive) {
  Clear();
  if (!ValidArchive(archive)) {
    return false;
  }
  if (storage_.mode == ::rund::replay::StorageMode::Memory) {
    static_cast<void>(Compact(archive.chunks));
  }
  loaded_diagnostic_ = std::move(archive.diagnostic);
  loaded_diagnostic_active_ = true;
  if (archive.records.empty() && archive.chunks.empty() &&
      archive.payload_hash == 0u) {
    return true;
  }
  ChunkLoad chunks = LoadChunks(archive, storage_);
  if (!chunks.ok()) {
    return false;
  }
  RecordLoad records = LoadRecords(archive);
  if (!records.ok()) {
    return false;
  }
  backend_->LoadArchive(std::move(chunks.blobs), std::move(chunks.refs),
                        chunks.encoded_bytes,
                        std::move(archive.spill_generation));
  records_ = std::move(records.records);
  pieces_ = std::move(records.pieces);
  RebuildRoleIndices();
  logical_bytes_ = records.logical_bytes;
  const bool hash_validated =
      ComputeRecordHashes() && payload_hash() == archive.payload_hash;
  loaded_payload_hash_ = archive.payload_hash;
  loaded_archive_ = true;
  return hash_validated;
}

} // namespace rund::node::replay_detail::payload
