#include "local.hpp"

#include "../backend.hpp"

#include <kernel/core/checked.hpp>

#include <limits>

namespace rund::node::replay_detail::payload {

namespace store_detail {

bool fits_budget(const std::uint64_t current, const std::uint64_t added,
                 const std::uint64_t budget) noexcept {
  std::uint64_t total = 0u;
  return rund::kernel::checked::add(current, added, total) && total <= budget;
}

bool fits_u32(const std::size_t value) noexcept {
  return value <= std::numeric_limits<std::uint32_t>::max();
}

} // namespace store_detail

std::optional<Limits> Limits::runtime(const std::uint32_t hosts,
                                      const std::uint32_t inputs,
                                      const std::uint64_t bytes) noexcept {
  const std::uint64_t records = static_cast<std::uint64_t>(hosts) + inputs;
  const std::uint64_t pieces = records + bytes / kChunkBytes;
  const std::uint64_t staged =
      bytes == 0u ? 0u : 1u + (bytes - 1u) / kChunkBytes;
  if (records > std::numeric_limits<std::uint32_t>::max() ||
      pieces > std::numeric_limits<std::uint32_t>::max() ||
      staged > std::numeric_limits<std::uint32_t>::max() ||
      bytes >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return Limits{.hosts = hosts,
                .inputs = inputs,
                .pieces = static_cast<std::size_t>(pieces),
                .blobs = static_cast<std::size_t>(pieces),
                .staged = static_cast<std::size_t>(staged),
                .bytes = bytes};
}

std::optional<Limits> Limits::archive(
    const ::rund::node::replay_detail::payload::Archive &archive) noexcept {
  Limits limits{};
  for (const ::rund::node::replay_detail::payload::ArchiveRecord &record :
       archive.records) {
    switch (record.metadata.role) {
    case ::rund::node::replay_detail::payload::Role::Host:
      ++limits.hosts;
      break;
    case ::rund::node::replay_detail::payload::Role::Input:
      ++limits.inputs;
      break;
    default:
      return std::nullopt;
    }
    if (record.pieces.size() >
        std::numeric_limits<std::size_t>::max() - limits.pieces) {
      return std::nullopt;
    }
    limits.pieces += record.pieces.size();
    if (!rund::kernel::checked::add(
            limits.bytes, record.metadata.completed_bytes, limits.bytes)) {
      return std::nullopt;
    }
  }
  limits.blobs = archive.chunks.size();
  if (!store_detail::fits_u32(limits.hosts + limits.inputs) ||
      !store_detail::fits_u32(limits.pieces) ||
      !store_detail::fits_u32(limits.blobs)) {
    return std::nullopt;
  }
  return limits;
}

} // namespace rund::node::replay_detail::payload
