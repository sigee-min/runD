#include "../internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace rund::measure::compute::route_matrix {
namespace {

[[nodiscard]] std::vector<std::string> split(const std::string_view line) {
  std::vector<std::string> fields;
  std::size_t begin = 0u;
  while (begin <= line.size()) {
    const std::size_t end = line.find(',', begin);
    fields.emplace_back(line.substr(begin, end == std::string_view::npos
                                               ? line.size() - begin
                                               : end - begin));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1u;
  }
  return fields;
}

[[nodiscard]] std::string key(const std::vector<std::string> &fields) {
  std::string result;
  constexpr std::array<std::size_t, 25u> KeyFields{
      3u,  4u,  5u,  6u,  7u,  8u,  9u,  10u, 11u, 12u, 13u,  14u, 15u,
      16u, 17u, 18u, 19u, 20u, 22u, 23u, 95u, 96u, 97u, 129u, 130u};
  for (const std::size_t index : KeyFields) {
    if (index >= fields.size()) {
      return {};
    }
    if (!result.empty()) {
      result.push_back('|');
    }
    result += fields[index];
  }
  return result;
}

[[nodiscard]] std::size_t
semantic_mismatch(const std::vector<std::string> &left,
                  const std::vector<std::string> &right) {
  if (left.size() != CsvFieldCount || right.size() != CsvFieldCount) {
    return CsvFieldCount;
  }
  for (std::size_t index = 3u; index < CsvFieldCount; ++index) {
    if (index == 21u || index == 28u || (index >= 33u && index <= 50u) ||
        index == 66u || index == 67u || index == 75u || index == 77u) {
      continue;
    }
    if (left[index] != right[index]) {
      return index;
    }
  }
  return CsvFieldCount;
}

[[nodiscard]] std::string_view column_name(const std::size_t index) noexcept {
  const std::string_view value = csv_header();
  std::size_t begin = 0u;
  for (std::size_t current = 0u; current <= index; ++current) {
    const std::size_t end = value.find(',', begin);
    if (current == index) {
      const std::size_t stop =
          end == std::string_view::npos ? value.find('\n', begin) : end;
      return stop == std::string_view::npos ? value.substr(begin)
                                            : value.substr(begin, stop - begin);
    }
    if (end == std::string_view::npos) {
      return {};
    }
    begin = end + 1u;
  }
  return {};
}

[[nodiscard]] bool
unique_keys(const std::vector<std::vector<std::string>> &packet_rows) {
  std::set<std::string> keys;
  for (const auto &fields : packet_rows) {
    const std::string row_key = key(fields);
    if (row_key.empty() || !keys.insert(row_key).second) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool parse_real(const std::string &value, double &result) {
  char *end = nullptr;
  const char *const begin = value.c_str();
  result = std::strtod(begin, &end);
  return end != begin && *end == '\0' && std::isfinite(result);
}

[[nodiscard]] bool parse_number(const std::string &value,
                                std::uint64_t &result) {
  try {
    std::size_t end = 0u;
    result = std::stoull(value, &end);
    return end == value.size();
  } catch (...) {
    return false;
  }
}

[[nodiscard]] bool median_real(
    const std::array<const std::vector<std::string> *, PacketCount> &rows,
    const std::size_t index, std::string &result) {
  std::array<double, PacketCount> values{};
  for (std::size_t packet = 0u; packet < PacketCount; ++packet) {
    if (!parse_real((*rows[packet])[index], values[packet])) {
      return false;
    }
  }
  std::sort(values.begin(), values.end());
  std::ostringstream text;
  text << std::fixed << std::setprecision(6) << values[PacketCount / 2u];
  result = text.str();
  return true;
}

[[nodiscard]] bool median_number(
    const std::array<const std::vector<std::string> *, PacketCount> &rows,
    const std::size_t index, std::string &result) {
  std::array<std::uint64_t, PacketCount> values{};
  for (std::size_t packet = 0u; packet < PacketCount; ++packet) {
    if (!parse_number((*rows[packet])[index], values[packet])) {
      return false;
    }
  }
  std::sort(values.begin(), values.end());
  result = std::to_string(values[PacketCount / 2u]);
  return true;
}

void hide_timing(std::vector<std::string> &fields,
                 const std::string_view authority) {
  for (std::size_t index = 33u; index <= 50u; ++index) {
    fields[index] = "0.000000";
  }
  fields[66u] = "0";
  fields[67u] = "0";
  fields[75u] = "0";
  fields[76u] = "unavailable_not_comparable";
  fields[77u] = "0";
  fields[78u] = "unavailable_not_comparable";
  fields[98u] = std::string{authority};
}

[[nodiscard]] std::string join(const std::vector<std::string> &fields) {
  std::ostringstream out;
  for (std::size_t index = 0u; index < fields.size(); ++index) {
    if (index != 0u) {
      out << ',';
    }
    out << fields[index];
  }
  return out.str();
}

[[nodiscard]] std::string error_row(const std::string_view reason) {
  std::vector<std::string> fields(CsvFieldCount);
  fields[0] = "route_matrix";
  fields[1] = "aggregate";
  fields[2] = "0";
  fields[3] = "unknown";
  fields[4] = "unsealed_current_source";
  fields[24] = "error";
  fields[26] = std::string{reason};
  fields[28] = "indeterminate";
  fields[98] = "unavailable_schema";
  return join(fields);
}

} // namespace

bool aggregate() {
  std::vector<std::vector<std::string>> packets[PacketCount + 1u];
  std::vector<std::string> raw;
  std::string line;
  bool schema_ok = true;
  while (std::getline(std::cin, line)) {
    if (line.rfind("route_matrix,", 0u) != 0u ||
        line.rfind("route_matrix,row_kind", 0u) == 0u) {
      continue;
    }
    auto fields = split(line);
    if (fields.size() != CsvFieldCount || fields[1] != "raw") {
      schema_ok = false;
      continue;
    }
    unsigned packet_number = 0u;
    try {
      packet_number = static_cast<unsigned>(std::stoul(fields[2]));
    } catch (...) {
      packet_number = 0u;
    }
    if (packet_number == 0u || packet_number > PacketCount ||
        key(fields).empty()) {
      schema_ok = false;
      continue;
    }
    raw.push_back(line);
    packets[packet_number].push_back(std::move(fields));
  }
  print_header();
  for (const std::string &value : raw) {
    std::fputs(value.c_str(), stdout);
    std::putchar('\n');
  }
  std::size_t expected_rows = 0u;
  std::string profile;
  if (!packets[1u].empty()) {
    profile = packets[1u][0u][3u];
    expected_rows = profile == "core" ? 4u : profile == "full" ? 18u : 0u;
  }
  for (unsigned packet_number = 1u; packet_number <= PacketCount;
       ++packet_number) {
    if (packets[packet_number].size() != expected_rows ||
        !unique_keys(packets[packet_number])) {
      schema_ok = false;
    }
    for (const auto &fields : packets[packet_number]) {
      if (fields[2u] != std::to_string(packet_number) ||
          fields[3u] != profile) {
        schema_ok = false;
      }
    }
  }
  if (!schema_ok || expected_rows == 0u) {
    const std::string failure = error_row("packet_schema_or_cardinality");
    std::fputs(failure.c_str(), stdout);
    std::putchar('\n');
    return false;
  }
  for (std::size_t index = 0u; index < expected_rows; ++index) {
    if (key(packets[1u][index]) != key(packets[2u][index]) ||
        key(packets[1u][index]) != key(packets[3u][index])) {
      const std::string failure = error_row("packet_order_or_route_drift");
      std::fputs(failure.c_str(), stdout);
      std::putchar('\n');
      return false;
    }
    const std::size_t mismatch_12 =
        semantic_mismatch(packets[1u][index], packets[2u][index]);
    const std::size_t mismatch_13 =
        semantic_mismatch(packets[1u][index], packets[3u][index]);
    if (mismatch_12 != CsvFieldCount || mismatch_13 != CsvFieldCount) {
      const std::size_t mismatch = std::min(mismatch_12, mismatch_13);
      std::string reason{"packet_semantic_drift_"};
      reason += column_name(mismatch);
      reason += "_";
      reason += std::to_string(mismatch);
      const std::string failure = error_row(reason);
      std::fputs(failure.c_str(), stdout);
      std::putchar('\n');
      return false;
    }
  }
  for (std::size_t index = 0u; index < expected_rows; ++index) {
    const std::array<const std::vector<std::string> *, PacketCount> rows{
        &packets[1u][index], &packets[2u][index], &packets[3u][index]};
    std::vector<std::string> aggregate_row = *rows[0u];
    aggregate_row[1] = "aggregate";
    aggregate_row[2] = "3";
    bool comparable = true;
    bool labels_match = (*rows[0u])[28u] == (*rows[1u])[28u] &&
                        (*rows[0u])[28u] == (*rows[2u])[28u];
    for (const auto *const packet_row : rows) {
      comparable = comparable && (*packet_row)[27u] == "1" &&
                   (*packet_row)[24u] == "ok" &&
                   (*packet_row)[98u] == "publishable";
    }
    const std::string consensus =
        comparable && labels_match && (*rows[0u])[28u] != "indeterminate"
            ? (*rows[0u])[28u]
            : "indeterminate";
    const bool published =
        comparable && labels_match && (*rows[0u])[28u] != "indeterminate";
    aggregate_row[24] = published ? "published" : "not_published";
    aggregate_row[25] = "0";
    aggregate_row[26] =
        published ? "three_packet_consensus" : "no_three_packet_consensus";
    aggregate_row[27] = published ? "1" : "0";
    aggregate_row[28] = consensus;
    if (published) {
      for (std::size_t field = 33u; field <= 50u; ++field) {
        if (!median_real(rows, field, aggregate_row[field])) {
          const std::string failure = error_row("timing_schema");
          std::fputs(failure.c_str(), stdout);
          std::putchar('\n');
          return false;
        }
      }
      for (const std::size_t field : {66u, 67u, 75u, 77u}) {
        if (!median_number(rows, field, aggregate_row[field])) {
          const std::string failure = error_row("timing_schema");
          std::fputs(failure.c_str(), stdout);
          std::putchar('\n');
          return false;
        }
      }
      aggregate_row[98u] = "published_median";
    } else {
      hide_timing(aggregate_row, comparable
                                     ? "unavailable_no_three_packet_consensus"
                                     : "unavailable_not_comparable");
    }
    std::string value = join(aggregate_row);
    std::fputs(value.c_str(), stdout);
    std::putchar('\n');
  }
  return true;
}

} // namespace rund::measure::compute::route_matrix
