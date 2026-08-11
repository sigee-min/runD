#pragma once

#include "model.hpp"

#include <cstdint>
#include <vector>

namespace rund::compute::detail::residency {

struct ByteRange final {
  std::uint32_t resource{};
  Access access{Access::Read};
  std::uint64_t offset{};
  std::uint64_t bytes{};
};

struct WindowFootprint final {
  std::uint32_t input_resource{};
  std::uint32_t output_resource{};
  std::uint64_t input_elements{};
  std::uint64_t output_first{};
  std::uint64_t output_count{};
  std::uint64_t window_size{};
  std::uint64_t stride{};
  std::uint64_t pad_left{};
  std::uint64_t element_bytes{};
};

struct ScanFootprint final {
  std::uint32_t input_resource{};
  std::uint32_t output_resource{};
  std::uint32_t partial_resource{};
  std::uint64_t element_count{};
  std::uint64_t tile_elements{};
  std::uint64_t element_bytes{};
};

[[nodiscard]] bool ProjectRange(const ByteRange &range,
                                std::uint64_t page_bytes,
                                std::vector<PageUse> &uses) noexcept;
[[nodiscard]] bool ProjectWindow(const WindowFootprint &footprint,
                                 std::uint64_t page_bytes,
                                 DemandEpoch &epoch) noexcept;
[[nodiscard]] bool ProjectScan(const ScanFootprint &footprint,
                               std::uint64_t page_bytes,
                               std::vector<DemandEpoch> &epochs) noexcept;

} // namespace rund::compute::detail::residency
