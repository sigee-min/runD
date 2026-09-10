#pragma once

#include "../schema.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace rund::measure::compute::virtual_crossover::aggregate_detail {

enum class CrossoverLabel : std::uint8_t {
  Invalid,
  Cpu,
  Metal,
  Indeterminate,
};

struct ConsensusCell final {
  std::array<CrossoverLabel, 3u> labels{};

  [[nodiscard]] constexpr CrossoverLabel published() const noexcept {
    return labels[0] != CrossoverLabel::Invalid && labels[0] == labels[1] &&
                   labels[1] == labels[2]
               ? labels[0]
               : CrossoverLabel::Invalid;
  }
};

struct CrossoverCsvSchema final {
  static constexpr std::size_t Missing =
      std::numeric_limits<std::size_t>::max();

  std::size_t field_count{};
  std::size_t n{Missing};
  std::size_t radius{Missing};
  std::size_t ratio_numerator{Missing};
  std::size_t ratio_denominator{Missing};
  std::size_t active_pages{Missing};
  std::size_t frame_capacity{Missing};
  std::size_t classification{Missing};
  std::size_t cpu_command_submits{Missing};
  std::size_t metal_command_submits{Missing};
  std::size_t cpu_window_handoffs{Missing};
  std::size_t cpu_window_batches{Missing};
  std::size_t cpu_window_queue_calls{Missing};
  std::size_t metal_window_handoffs{Missing};
  std::size_t metal_window_batches{Missing};
  std::size_t metal_window_queue_calls{Missing};

  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] bool operator==(const CrossoverCsvSchema &) const noexcept;
};

using CsvFields = std::array<std::string_view, 96u>;

[[nodiscard]] std::string_view label_name(CrossoverLabel) noexcept;
[[nodiscard]] CrossoverLabel parse_label(std::string_view) noexcept;
[[nodiscard]] bool parse_size(std::string_view, std::size_t &) noexcept;
[[nodiscard]] bool split_csv(std::string_view, CsvFields &,
                             std::size_t &) noexcept;
[[nodiscard]] CrossoverCsvSchema csv_schema(const CsvFields &,
                                            std::size_t) noexcept;
[[nodiscard]] bool expected_key(std::size_t, const CsvFields &, std::size_t,
                                CrossoverCsvSchema) noexcept;

void print_consensus(
    const std::array<ConsensusCell, virtual_crossover::CellCount> &) noexcept;

} // namespace rund::measure::compute::virtual_crossover::aggregate_detail
