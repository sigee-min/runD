#include "model.hpp"

#include "src/runtime/replay/host/payload/hash.hpp"

#include <utility>

namespace replay_payload_store {

std::vector<std::byte> Payload(const std::string_view text) {
  std::vector<std::byte> bytes{};
  bytes.reserve(text.size());
  for (const char value : text) {
    bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(value)));
  }
  return bytes;
}

Store Prepared(const std::uint32_t hosts, const std::uint32_t inputs,
               const std::uint64_t bytes, ::rund::replay::Storage storage,
               const ::rund::replay::Diagnostic diagnostic) {
  const auto limits =
      rund::node::replay_detail::payload::Limits::runtime(hosts, inputs, bytes);
  return Store{std::move(storage), limits.value(), diagnostic};
}

std::uint64_t Identity(const Archive &archive) {
  rund::node::replay_detail::payload::ArchiveHash archive_hash{
      static_cast<std::uint64_t>(archive.records.size())};
  for (const auto &record : archive.records) {
    rund::node::replay_detail::payload::RecordHash record_hash{
        record.metadata, record.metadata.completed_bytes,
        static_cast<std::uint64_t>(record.pieces.size())};
    for (const auto &piece : record.pieces) {
      const auto &chunk =
          archive.chunks[static_cast<std::size_t>(piece.chunk_id)];
      record_hash.Append(chunk.uncompressed_hash, chunk.uncompressed_bytes);
    }
    archive_hash.Append(record_hash.Finish());
  }
  return archive_hash.Finish();
}

int Run(const std::span<const Contract> contracts) {
  for (const Contract contract : contracts) {
    if (const int result = contract(); result != 0) {
      return result;
    }
  }
  return 0;
}

} // namespace replay_payload_store
