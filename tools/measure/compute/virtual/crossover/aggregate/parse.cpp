#include "local.hpp"

#include <charconv>

namespace rund::measure::compute::virtual_crossover::aggregate_detail {

bool parse_size(const std::string_view value, std::size_t &result) noexcept {
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

bool split_csv(std::string_view line, CsvFields &fields,
               std::size_t &field_count) noexcept {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
    line.remove_suffix(1u);
  }
  field_count = 0u;
  std::size_t begin = 0u;
  while (begin <= line.size()) {
    if (field_count == fields.size()) {
      return false;
    }
    const std::size_t comma = line.find(',', begin);
    if (comma == std::string_view::npos) {
      fields[field_count++] = line.substr(begin);
      return true;
    }
    fields[field_count++] = line.substr(begin, comma - begin);
    begin = comma + 1u;
  }
  return true;
}

bool CrossoverCsvSchema::valid() const noexcept {
  return field_count != 0u && n < field_count && radius < field_count &&
         ratio_numerator < field_count && ratio_denominator < field_count &&
         active_pages < field_count && frame_capacity < field_count &&
         classification < field_count && cpu_command_submits < field_count &&
         metal_command_submits < field_count &&
         cpu_window_handoffs < field_count &&
         cpu_window_batches < field_count &&
         cpu_window_queue_calls < field_count &&
         metal_window_handoffs < field_count &&
         metal_window_batches < field_count &&
         metal_window_queue_calls < field_count;
}

bool CrossoverCsvSchema::operator==(const CrossoverCsvSchema &) const noexcept =
    default;

namespace {

[[nodiscard]] std::size_t find_field(const CsvFields &fields,
                                     const std::size_t field_count,
                                     const std::string_view name) noexcept {
  std::size_t found = CrossoverCsvSchema::Missing;
  for (std::size_t index = 1u; index < field_count; ++index) {
    if (fields[index] != name) {
      continue;
    }
    if (found != CrossoverCsvSchema::Missing) {
      return CrossoverCsvSchema::Missing;
    }
    found = index;
  }
  return found;
}

} // namespace

CrossoverCsvSchema csv_schema(const CsvFields &fields,
                              const std::size_t field_count) noexcept {
  if (field_count == 0u || fields[0] != "virtual_crossover_columns") {
    return {};
  }
  return CrossoverCsvSchema{
      .field_count = field_count,
      .n = find_field(fields, field_count, "n"),
      .radius = find_field(fields, field_count, "radius"),
      .ratio_numerator = find_field(fields, field_count, "active_ratio_num"),
      .ratio_denominator = find_field(fields, field_count, "active_ratio_den"),
      .active_pages = find_field(fields, field_count, "active_pages"),
      .frame_capacity = find_field(fields, field_count, "frame_capacity"),
      .classification =
          find_field(fields, field_count, "robust_classification"),
      .cpu_command_submits =
          find_field(fields, field_count, "cpu_command_submits"),
      .metal_command_submits =
          find_field(fields, field_count, "metal_command_submits"),
      .cpu_window_handoffs =
          find_field(fields, field_count, "cpu_window_handoffs"),
      .cpu_window_batches =
          find_field(fields, field_count, "cpu_window_batches"),
      .cpu_window_queue_calls =
          find_field(fields, field_count, "cpu_window_queue_calls"),
      .metal_window_handoffs =
          find_field(fields, field_count, "metal_window_handoffs"),
      .metal_window_batches =
          find_field(fields, field_count, "metal_window_batches"),
      .metal_window_queue_calls =
          find_field(fields, field_count, "metal_window_queue_calls"),
  };
}

bool expected_key(const std::size_t ordinal, const CsvFields &fields,
                  const std::size_t field_count,
                  const CrossoverCsvSchema schema) noexcept {
  if (ordinal >= ::rund::measure::compute::virtual_crossover::CellCount ||
      !schema.valid() || field_count != schema.field_count ||
      fields[1] != "current_source_diagnostic" || fields[2] != "ok") {
    return false;
  }
  const std::size_t n_index =
      ordinal /
      (::rund::measure::compute::virtual_crossover::Radii.size() *
       ::rund::measure::compute::virtual_crossover::ActiveRatios.size());
  const std::size_t radius_index =
      ordinal /
      ::rund::measure::compute::virtual_crossover::ActiveRatios.size() %
      ::rund::measure::compute::virtual_crossover::Radii.size();
  const std::size_t ratio_index =
      ordinal %
      ::rund::measure::compute::virtual_crossover::ActiveRatios.size();
  std::size_t n = 0u;
  std::size_t radius = 0u;
  std::size_t numerator = 0u;
  std::size_t denominator = 0u;
  std::size_t active_pages = 0u;
  std::size_t frame_capacity = 0u;
  std::size_t cpu_submits = 0u;
  std::size_t metal_submits = 0u;
  std::size_t cpu_handoffs = 0u;
  std::size_t cpu_batches = 0u;
  std::size_t cpu_queue_calls = 0u;
  std::size_t metal_handoffs = 0u;
  std::size_t metal_batches = 0u;
  std::size_t metal_queue_calls = 0u;
  return parse_size(fields[schema.n], n) &&
         parse_size(fields[schema.radius], radius) &&
         parse_size(fields[schema.ratio_numerator], numerator) &&
         parse_size(fields[schema.ratio_denominator], denominator) &&
         parse_size(fields[schema.active_pages], active_pages) &&
         parse_size(fields[schema.frame_capacity], frame_capacity) &&
         parse_size(fields[schema.cpu_command_submits], cpu_submits) &&
         parse_size(fields[schema.metal_command_submits], metal_submits) &&
         parse_size(fields[schema.cpu_window_handoffs], cpu_handoffs) &&
         parse_size(fields[schema.cpu_window_batches], cpu_batches) &&
         parse_size(fields[schema.cpu_window_queue_calls], cpu_queue_calls) &&
         parse_size(fields[schema.metal_window_handoffs], metal_handoffs) &&
         parse_size(fields[schema.metal_window_batches], metal_batches) &&
         parse_size(fields[schema.metal_window_queue_calls],
                    metal_queue_calls) &&
         n == ::rund::measure::compute::virtual_crossover::LogicalCounts
                  [n_index] &&
         radius ==
             ::rund::measure::compute::virtual_crossover::Radii[radius_index] &&
         numerator == ::rund::measure::compute::virtual_crossover::ActiveRatios
                          [ratio_index]
                              .numerator &&
         denominator == ::rund::measure::compute::virtual_crossover::
                            ActiveRatios[ratio_index]
                                .denominator &&
         frame_capacity != 0u && cpu_submits == 0u && cpu_handoffs == 0u &&
         cpu_batches == 0u && cpu_queue_calls == 0u &&
         metal_submits ==
             ::rund::measure::compute::virtual_crossover::divide_up(
                 active_pages, frame_capacity) &&
         metal_handoffs == 0u && metal_batches == 0u && metal_queue_calls == 0u;
}

CrossoverLabel parse_label(const std::string_view value) noexcept {
  return value == "cpu_observed"     ? CrossoverLabel::Cpu
         : value == "metal_observed" ? CrossoverLabel::Metal
         : value == "indeterminate"  ? CrossoverLabel::Indeterminate
                                     : CrossoverLabel::Invalid;
}

} // namespace rund::measure::compute::virtual_crossover::aggregate_detail
