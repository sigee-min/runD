#include "../crossover.hpp"
#include "aggregate/local.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <string_view>

namespace rund::measure::compute {
namespace {

namespace vc = ::rund::measure::compute::virtual_crossover;
namespace aggregate = vc::aggregate_detail;

} // namespace

bool AggregateVirtualCrossover() {
  std::array<aggregate::ConsensusCell, vc::CellCount> cells{};
  std::array<std::size_t, 3u> cell_counts{};
  std::array<bool, 3u> packet_complete{};
  std::array<aggregate::CrossoverCsvSchema, 3u> schemas{};
  std::array<char, 8'192u> line{};
  aggregate::CsvFields fields{};
  std::size_t field_count = 0u;
  std::size_t next_packet = 0u;
  std::size_t active_packet = 3u;
  bool valid = true;

  while (std::fgets(line.data(), static_cast<int>(line.size()), stdin) !=
         nullptr) {
    const std::string_view raw{line.data()};
    std::fputs(line.data(), stdout);
    if (raw.empty() || raw.back() != '\n' ||
        !aggregate::split_csv(raw, fields, field_count)) {
      valid = false;
      continue;
    }
    if (fields[0] == "diagnostic_packet") {
      std::size_t ordinal = 0u;
      std::size_t total = 0u;
      if (field_count != 4u || !aggregate::parse_size(fields[2], ordinal) ||
          !aggregate::parse_size(fields[3], total) || total != 3u ||
          ordinal == 0u || ordinal > 3u) {
        valid = false;
        continue;
      }
      const std::size_t packet = ordinal - 1u;
      if (fields[1] == "begin") {
        if (active_packet != 3u || packet != next_packet ||
            packet_complete[packet]) {
          valid = false;
        } else {
          active_packet = packet;
        }
      } else if (fields[1] == "end") {
        if (active_packet != packet || cell_counts[packet] != vc::CellCount) {
          valid = false;
        } else {
          packet_complete[packet] = true;
          active_packet = 3u;
          ++next_packet;
        }
      } else {
        valid = false;
      }
      continue;
    }
    if (fields[0] == "virtual_crossover_columns") {
      const aggregate::CrossoverCsvSchema schema =
          aggregate::csv_schema(fields, field_count);
      if (active_packet >= 3u || schemas[active_packet].valid() ||
          !schema.valid() || (active_packet != 0u && schema != schemas[0])) {
        valid = false;
      } else {
        schemas[active_packet] = schema;
      }
      continue;
    }
    if (fields[0] != "virtual_crossover") {
      continue;
    }
    if (active_packet >= 3u || !schemas[active_packet].valid() ||
        !aggregate::expected_key(cell_counts[active_packet], fields,
                                 field_count, schemas[active_packet])) {
      valid = false;
      continue;
    }
    const aggregate::CrossoverLabel label =
        aggregate::parse_label(fields[schemas[active_packet].classification]);
    if (label == aggregate::CrossoverLabel::Invalid) {
      valid = false;
      continue;
    }
    cells[cell_counts[active_packet]].labels[active_packet] = label;
    ++cell_counts[active_packet];
  }

  if (active_packet != 3u || next_packet != 3u ||
      !std::all_of(packet_complete.begin(), packet_complete.end(),
                   [](const bool complete) { return complete; }) ||
      !std::all_of(schemas.begin(), schemas.end(),
                   [](const aggregate::CrossoverCsvSchema schema) {
                     return schema.valid();
                   })) {
    valid = false;
  }
  if (!valid) {
    std::fputs("virtual_crossover_consensus,error,invalid_packet_stream\n",
               stderr);
    return false;
  }
  aggregate::print_consensus(cells);
  return true;
}

} // namespace rund::measure::compute
