#pragma once

#include <kernel/core/checked.hpp>
#include "../../backend.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace rund::node::replay_detail::payload::spill {

inline constexpr std::uint64_t kRecordMagic = 0x72644853504c3031ull;
inline constexpr std::uint64_t kHeaderBytes =
    sizeof(std::uint64_t) * 5u + sizeof(std::uint8_t);

struct Space final {
  std::uint64_t allocation = 0u;
  bool headroom = false;
};

struct Segment final {
  ::rund::replay::Code code = ::rund::replay::Code::HostPayloadNotLoaded;
  Blob blob{};

  [[nodiscard]] bool ok() const noexcept {
    return code == ::rund::replay::Code::Ok;
  }
};

[[nodiscard]] std::optional<Space>
inspect(const std::string &directory, std::uint64_t segment_bytes,
        std::uint64_t record_bytes, std::uint64_t minimum_free_bytes) noexcept;

[[nodiscard]] std::string
path(const std::shared_ptr<const SpillGeneration> &generation,
     std::uint32_t segment_index);

[[nodiscard]] bool
write(const std::shared_ptr<const SpillGeneration> &generation,
      std::uint32_t segment_index, std::uint64_t segment_offset,
      std::uint32_t blob_index, const Blob &blob) noexcept;

[[nodiscard]] Segment
read(const std::shared_ptr<const SpillGeneration> &generation,
     std::span<const SpillRef> refs, std::span<const Blob> blobs,
     std::uint32_t blob_index);

} // namespace rund::node::replay_detail::payload::spill
