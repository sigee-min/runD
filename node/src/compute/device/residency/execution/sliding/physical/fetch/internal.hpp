#pragma once

#include "../../internal.hpp"

namespace rund::compute::detail::residency::physical::fetch {

struct Projection final {
  execution::FetchSource source{};
  execution::FetchReuseSource reuse{};
  CacheUse demand{};
  FrameRegion region{};
  std::array<FrameRegion, execution::BankCapacity> host_regions{};
  std::uint64_t expected_bytes{};
};

struct Selection final {
  std::size_t slot{};
  std::uint32_t reuse_frame{};
  bool hit{};
  bool reuses_frame{};
};

struct Commit final {
  std::uint64_t turn{};
  std::uint32_t physical_frame{};
  std::uint32_t reuse_frame{};
  bool reuses_frame{};
};

} // namespace rund::compute::detail::residency::physical::fetch
