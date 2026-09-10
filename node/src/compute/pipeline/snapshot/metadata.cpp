#include "internal.hpp"

#include "../../../hash/fnv.hpp"
#include "../../size.hpp"

namespace rund::compute::detail::snapshot_detail {

Status shape(const PipelinePublicationState &publication,
             std::size_t &bytes) noexcept {
  if (publication.state_pairs.empty()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  if (publication.state_pairs.size() > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineCapacity);
  }
  bytes = 0u;
  for (const PipelineStatePair &pair : publication.state_pairs) {
    if (!size::add(bytes, pair.bytes, bytes)) {
      return Status::fail(Reason::BufferCapacity);
    }
  }
  return Status::success();
}

Status prepare_metadata(const PipelinePublicationState &publication,
                        const std::uint64_t generation,
                        const std::size_t byte_capacity,
                        const std::size_t field_capacity,
                        StateSnapshotState &snapshot) {
  std::size_t bytes = 0u;
  const Status shaped = shape(publication, bytes);
  if (!shaped) {
    return shaped;
  }
  if (bytes > byte_capacity ||
      publication.state_pairs.size() > field_capacity ||
      snapshot.fields.capacity() < publication.state_pairs.size()) {
    return Status::fail(Reason::BufferCapacity);
  }
  snapshot.fields.clear();
  snapshot.fingerprint = publication.fingerprint;
  snapshot.generation = generation;
  snapshot.byte_count = bytes;
  snapshot.hash = 0u;
  std::size_t offset = 0u;
  for (const PipelineStatePair &pair : publication.state_pairs) {
    snapshot.fields.push_back(PipelineSnapshotField{
        .type = pair.type,
        .format = pair.format,
        .count = pair.count,
        .offset = offset,
        .bytes = pair.bytes,
        .payload_hash = ::rund::node::hash_detail::kFnvOffset,
    });
    offset += pair.bytes;
  }
  return Status::success();
}

} // namespace rund::compute::detail::snapshot_detail
