#include "materialize.hpp"
#include "chunk.hpp"
#include "hash.hpp"
#include <kernel/core/checked.hpp>

#include <algorithm>
#include <utility>

namespace rund::node::replay_detail::payload {
namespace {

[[nodiscard]] std::uint64_t
hash_record(const MaterializedRecord &record) noexcept {
  const std::size_t chunk_count =
      record.bytes.empty() ? 0u : 1u + (record.bytes.size() - 1u) / kChunkBytes;
  RecordHash hash{record.metadata,
                  static_cast<std::uint64_t>(record.bytes.size()),
                  static_cast<std::uint64_t>(chunk_count)};
  for (std::size_t offset = 0u; offset < record.bytes.size();
       offset += kChunkBytes) {
    const std::size_t size =
        std::min(kChunkBytes, record.bytes.size() - offset);
    const std::span<const std::byte> chunk{record.bytes.data() + offset, size};
    const ::rund::StableHash chunk_hash =
        chunk_count == 1u
            ? record.metadata.payload_hash
            : ::rund::host::hash_bytes(chunk.data(), chunk.size());
    hash.Append(chunk_hash, static_cast<std::uint64_t>(size));
  }
  return hash.Finish();
}

} // namespace

Materialization Materialize(std::vector<MaterializedRecord> records) {
  ArchiveHash hash{static_cast<std::uint64_t>(records.size())};
  for (const MaterializedRecord &record : records) {
    hash.Append(hash_record(record));
  }
  return Materialization{.records = std::move(records),
                         .payload_hash = hash.Finish()};
}

bool Equal(const Materialization &expected,
           const Materialization &actual) noexcept {
  if (expected.payload_hash != actual.payload_hash ||
      expected.records.size() != actual.records.size()) {
    return false;
  }
  for (std::size_t index = 0u; index < expected.records.size(); ++index) {
    const MaterializedRecord &left = expected.records[index];
    const MaterializedRecord &right = actual.records[index];
    if (left.metadata != right.metadata || left.bytes != right.bytes) {
      return false;
    }
  }
  return true;
}

::rund::node::replay_detail::payload::Archive
MakeArchive(const Materialization &materialized) {
  BuildResult built = Build(materialized);
  if (!built.ok()) {
    return ::rund::node::replay_detail::payload::Archive{};
  }
  return built.store.Archive();
}

BuildResult Build(const Materialization &materialized) {
  return Build(materialized, ::rund::replay::Storage{});
}

BuildResult Build(const Materialization &materialized,
                  ::rund::replay::Storage storage) {
  std::uint32_t hosts = 0u;
  std::uint32_t inputs = 0u;
  std::uint64_t bytes = 0u;
  for (const MaterializedRecord &record : materialized.records) {
    std::uint32_t &count =
        record.metadata.role ==
                ::rund::node::replay_detail::payload::Role::Input
            ? inputs
            : hosts;
    if (count == std::numeric_limits<std::uint32_t>::max() ||
        !rund::kernel::checked::add(bytes, record.metadata.completed_bytes,
                                    bytes)) {
      return BuildResult{.code =
                             ::rund::replay::Code::HostPayloadCapacityExceeded};
    }
    ++count;
  }
  const std::optional<Limits> limits = Limits::runtime(hosts, inputs, bytes);
  if (!limits.has_value() || bytes > storage.max_bytes) {
    return BuildResult{.code =
                           ::rund::replay::Code::HostPayloadCapacityExceeded};
  }
  Store store{std::move(storage), *limits};
  if (materialized.records.empty() && materialized.payload_hash == 0u) {
    return BuildResult{.code = ::rund::replay::Code::Ok,
                       .store = std::move(store)};
  }
  for (const MaterializedRecord &record : materialized.records) {
    bool appended = false;
    if (record.metadata.role ==
        ::rund::node::replay_detail::payload::Role::Input) {
      ::rund::node::replay_detail::payload::Bytes bytes =
          ::rund::node::replay_detail::payload::Bytes::freeze(
              std::vector<std::byte>{record.bytes.begin(), record.bytes.end()});
      appended = store.AppendInput(
          record.metadata.input_source, record.metadata.input_schema,
          record.metadata.input_sequence,
          InputSourceRange{
              .event_offset = record.metadata.source_event_offset,
              .event_count = record.metadata.source_event_count,
              .payload_offset = record.metadata.source_payload_offset,
              .payload_count = record.metadata.source_payload_count,
              .hash = record.metadata.source_hash,
          },
          bytes, Capture::verify(bytes.span(), record.metadata.payload_hash));
    } else {
      appended = store.Append(
          record.metadata.event_sequence, record.metadata.kind,
          Capture::verify(record.bytes, record.metadata.payload_hash));
    }
    if (!appended) {
      return BuildResult{.code = ::rund::replay::Code::HostPayloadHashInvalid,
                         .store = Store{}};
    }
  }
  if (store.payload_hash() != materialized.payload_hash) {
    return BuildResult{.code = ::rund::replay::Code::HostPayloadHashInvalid,
                       .store = Store{}};
  }
  return BuildResult{.code = ::rund::replay::Code::Ok,
                     .store = std::move(store)};
}

Materialization Materialize(const Store &store) {
  std::vector<MaterializedRecord> records{};
  records.reserve(store.records().size());
  for (std::size_t index = 0u; index < store.records().size(); ++index) {
    const StoredRecord &stored = store.records()[index];
    ResolveResult resolved = store.Resolve(index);
    records.push_back(MaterializedRecord{
        .metadata = stored.metadata,
        .bytes = resolved.ok()
                     ? std::vector<std::byte>{resolved.bytes.span().begin(),
                                              resolved.bytes.span().end()}
                     : std::vector<std::byte>{}});
  }
  return Materialize(std::move(records));
}

} // namespace rund::node::replay_detail::payload
