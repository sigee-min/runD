#pragma once

#include "../stride.hpp"

#include <kernel/program/compute/limit.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace rund::node::accel::detail {

inline constexpr std::size_t MapSpecializationEditCapacity =
    3u * static_cast<std::size_t>(rund::kernel::kMaxComputeBindingCount);

struct MapSourceEdit final {
  std::size_t begin{};
  std::size_t end{};
  std::array<char, 20u> replacement{};
  std::uint8_t replacement_size{};

  [[nodiscard]] std::string_view text() const noexcept {
    return std::string_view{replacement.data(), replacement_size};
  }
};

struct MapSourceSpecialization final {
  std::array<MapSourceEdit, MapSpecializationEditCapacity> edits{};
  std::size_t edit_count{};
  std::uint64_t exact_source_bytes{};
  std::uint64_t source_upper_bytes{};
  std::uint64_t reserve_upper_bytes{};
  bool ok{};
  const char *reason{"compute_artifact_mismatch"};

  [[nodiscard]] std::span<const MapSourceEdit> active_edits() const noexcept {
    return {edits.data(), edit_count};
  }
};

[[nodiscard]] MapSourceSpecialization PlanMapSourceSpecialization(
    const rund::kernel::LoweringArtifact &, const rund::kernel::ComputePlan &,
    const rund::kernel::BindingSet &, std::uint64_t alignment,
    std::uint64_t reserve_upper) noexcept;

} // namespace rund::node::accel::detail
