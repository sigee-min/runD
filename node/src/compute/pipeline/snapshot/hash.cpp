#include "internal.hpp"

#include "../../../hash/fnv.hpp"
#include "../../size.hpp"
#include "../../type.hpp"

#include <algorithm>

namespace rund::compute::detail::snapshot_detail {
namespace {

void mix(std::uint64_t &value, const std::uint64_t part) noexcept {
  ::rund::node::hash_detail::MixU64(value, part);
}

} // namespace

std::uint64_t hash(const StateSnapshotState &snapshot) noexcept {
  std::uint64_t value = ::rund::node::hash_detail::kFnvOffset;
  mix(value, snapshot.fingerprint.hi);
  mix(value, snapshot.fingerprint.lo);
  mix(value, snapshot.generation);
  mix(value, snapshot.fields.size());
  for (const PipelineSnapshotField &field : snapshot.fields) {
    mix(value, static_cast<std::uint64_t>(field.type));
    mix(value, field.format.integer_bits);
    mix(value, field.format.fraction_bits);
    mix(value, static_cast<std::uint64_t>(field.format.rounding));
    mix(value, static_cast<std::uint64_t>(field.format.overflow));
    mix(value, static_cast<std::uint64_t>(field.format.approximation));
    mix(value, field.count);
    mix(value, field.offset);
    mix(value, field.bytes);
    mix(value, field.payload_hash);
  }
  return value;
}

bool valid_layout(const StateSnapshotState &snapshot) noexcept {
  if (!snapshot.fingerprint || snapshot.fields.empty() ||
      (snapshot.byte_count != 0u && snapshot.bytes == nullptr)) {
    return false;
  }
  std::size_t offset = 0u;
  for (const PipelineSnapshotField &field : snapshot.fields) {
    const std::size_t element_bytes = type_bytes(field.type);
    std::size_t field_bytes = 0u;
    if (element_bytes == 0u || field.offset != offset ||
        !size::multiply(field.count, element_bytes, field_bytes) ||
        field.bytes != field_bytes ||
        field.bytes >
            snapshot.byte_count - std::min(snapshot.byte_count, offset) ||
        field.payload_hash !=
            (field.bytes == 0u
                 ? ::rund::node::hash_detail::kFnvOffset
                 : ::rund::node::hash_detail::HashBytes(
                       snapshot.bytes.get() + field.offset, field.bytes))) {
      return false;
    }
    if (!size::add(offset, field.bytes, offset)) {
      return false;
    }
  }
  return offset == snapshot.byte_count && snapshot.hash == hash(snapshot);
}

} // namespace rund::compute::detail::snapshot_detail
