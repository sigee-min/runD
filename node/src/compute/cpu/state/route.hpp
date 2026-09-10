#pragma once

#include <cstddef>
#include <cstdint>

namespace rund::compute::detail {

struct CpuGraphProgram;
struct CpuRuntimeGraph;

struct CpuRunRoutePlan final {
  const CpuGraphProgram *program = nullptr;
  const CpuRuntimeGraph *runtime = nullptr;
  std::uint64_t graph_hash{};
  std::size_t step_count{};
  std::size_t map_count{};
  std::size_t read_count{};
  std::size_t write_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuRunRoutePlan &) const noexcept = default;
};

struct CpuRunRouteSlice final {
  std::size_t map_begin{};
  std::size_t map_count{};
  std::size_t read_begin{};
  std::size_t read_count{};
  std::size_t write_begin{};
  std::size_t write_count{};

  [[nodiscard]] constexpr bool
  operator==(const CpuRunRouteSlice &) const noexcept = default;
};

} // namespace rund::compute::detail
