#pragma once

#include <kernel/core/checked.hpp>
#include "../../backend.hpp"
#include "../../store.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace rund::node::replay_detail::payload {

struct ChunkLoad {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  std::vector<Blob> blobs{};
  std::vector<SpillRef> refs{};
  std::uint64_t encoded_bytes = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

struct RecordLoad {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  std::vector<StoredRecord> records{};
  std::vector<Piece> pieces{};
  std::uint64_t logical_bytes = 0u;

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

[[nodiscard]] ChunkLoad
LoadChunks(::rund::node::replay_detail::payload::Archive &archive,
           const ::rund::replay::Storage &storage);

[[nodiscard]] RecordLoad
LoadRecords(const ::rund::node::replay_detail::payload::Archive &archive);

} // namespace rund::node::replay_detail::payload
