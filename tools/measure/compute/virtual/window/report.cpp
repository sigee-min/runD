#include "../window.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <string_view>

namespace rund::measure::compute {
namespace virtual_window {
namespace {

[[nodiscard]] bool split_csv(std::string_view line,
                             std::array<std::string_view, 96u> &fields,
                             std::size_t &count) noexcept {
  while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
    line.remove_suffix(1u);
  }
  count = 0u;
  std::size_t begin = 0u;
  while (begin <= line.size()) {
    if (count == fields.size()) {
      return false;
    }
    const std::size_t comma = line.find(',', begin);
    if (comma == std::string_view::npos) {
      fields[count++] = line.substr(begin);
      return true;
    }
    fields[count++] = line.substr(begin, comma - begin);
    begin = comma + 1u;
  }
  return true;
}

[[nodiscard]] bool parse_size(const std::string_view value,
                              std::size_t &result) noexcept {
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  return parsed.ec == std::errc{} && parsed.ptr == value.data() + value.size();
}

[[nodiscard]] constexpr bool valid_label(const std::string_view value) {
  return value == "cpu_observed" || value == "metal_observed" ||
         value == "indeterminate";
}

} // namespace
} // namespace virtual_window

void PrintVirtualWindowColumns() {
  std::fputs(
      "virtual_window_columns,evidence_scope,status,q,logical_capacity,"
      "active_count,page_elements,active_pages,frame_capacity,route_shape,"
      "receipt_scope,graph_hash,cpu_plan_hi,cpu_plan_lo,metal_plan_hi,"
      "metal_plan_lo,cpu_output_hash,metal_output_hash,cpu_p25_us,cpu_p50_us,"
      "cpu_p75_us,cpu_p95_us,cpu_mad_us,metal_p25_us,metal_p50_us,"
      "metal_p75_us,metal_p95_us,metal_mad_us,metal_speedup,classification,"
      "cpu_epochs,cpu_dispatches,cpu_command_submits,cpu_inflight_peak,"
      "cpu_window_handoffs,cpu_window_batches,cpu_window_queue_calls,"
      "cpu_h2d_submits,cpu_d2h_submits,cpu_uploaded_bytes,"
      "cpu_downloaded_bytes,cpu_stall_ns,cpu_overlap_ns,cpu_h2d_overlap_ns,"
      "cpu_d2h_overlap_ns,cpu_backing_read_bytes,cpu_backing_write_bytes,"
      "cpu_page_in,cpu_cache_hits,cpu_page_out,cpu_page_in_bytes,"
      "cpu_page_out_bytes,cpu_submit_wait_ns,cpu_readback_ns,cpu_kernel_ns,"
      "cpu_kernel_samples,cpu_sampled,cpu_allocfree,metal_epochs,"
      "metal_dispatches,metal_command_submits,metal_inflight_peak,"
      "metal_window_handoffs,metal_window_batches,metal_window_queue_calls,"
      "metal_h2d_submits,metal_d2h_submits,metal_uploaded_bytes,"
      "metal_downloaded_bytes,metal_stall_ns,metal_overlap_ns,"
      "metal_h2d_overlap_ns,metal_d2h_overlap_ns,metal_backing_read_bytes,"
      "metal_backing_write_bytes,metal_page_in,metal_cache_hits,"
      "metal_page_out,metal_page_in_bytes,metal_page_out_bytes,"
      "metal_submit_wait_ns,metal_readback_ns,metal_kernel_ns,"
      "metal_kernel_samples,metal_sampled,metal_allocfree,terminal_reads,"
      "profile_projections\n",
      stdout);
}

bool AggregateVirtualWindow() {
  std::array<std::array<std::string_view, 3u>, 3u> labels{};
  std::array<std::array<char, 32u>, 9u> label_storage{};
  std::array<std::size_t, 3u> rows{};
  std::array<bool, 3u> schemas{};
  std::array<char, 32'768u> buffer{};
  std::array<std::string_view, 96u> fields{};
  std::size_t packet = std::numeric_limits<std::size_t>::max();
  std::size_t storage = 0u;
  bool valid = true;
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), stdin) !=
         nullptr) {
    std::size_t count = 0u;
    if (!virtual_window::split_csv(buffer.data(), fields, count) ||
        count == 0u) {
      valid = false;
      continue;
    }
    if (fields[0] == "diagnostic_packet") {
      std::size_t ordinal = 0u;
      if (count != 4u || !virtual_window::parse_size(fields[2], ordinal) ||
          ordinal == 0u || ordinal > 3u || fields[3] != "3") {
        valid = false;
        continue;
      }
      if (fields[1] == "begin" &&
          packet == std::numeric_limits<std::size_t>::max()) {
        packet = ordinal - 1u;
      } else if (fields[1] == "end" && packet == ordinal - 1u) {
        packet = std::numeric_limits<std::size_t>::max();
      } else {
        valid = false;
      }
      continue;
    }
    if (fields[0] == "virtual_window_columns") {
      if (packet >= schemas.size() || schemas[packet] || count != 88u ||
          fields[3] != "q" || fields[9] != "route_shape" ||
          fields[10] != "receipt_scope" || fields[29] != "classification" ||
          fields[34] != "cpu_window_handoffs" ||
          fields[62] != "metal_window_handoffs" ||
          fields[86] != "terminal_reads" ||
          fields[87] != "profile_projections") {
        valid = false;
      } else {
        schemas[packet] = true;
      }
      continue;
    }
    if (fields[0] != "virtual_window") {
      continue;
    }
    std::size_t q = 0u;
    std::size_t cpu_submits = 0u;
    std::size_t metal_submits = 0u;
    if (packet >= rows.size() || !schemas[packet] || count != 88u ||
        fields[2] != "ok" || !virtual_window::parse_size(fields[3], q) ||
        !virtual_window::parse_size(fields[32], cpu_submits) ||
        !virtual_window::parse_size(fields[60], metal_submits) ||
        fields[9] != "direct_pointwise_q2_4_coherent_required" ||
        fields[10] != "public_window_receipt" || rows[packet] >= 3u ||
        fields[3] !=
            std::array<std::string_view, 3u>{"2", "3", "4"}[rows[packet]] ||
        cpu_submits != 0u || metal_submits != q || fields[30] != fields[3] ||
        fields[34] != "0" || fields[35] != "0" || fields[36] != "0" ||
        fields[58] != fields[3] || fields[62] != "1" ||
        fields[63] != fields[3] || fields[64] != fields[3] ||
        fields[56] != "60" || fields[57] != "60" || fields[84] != "60" ||
        fields[85] != "60" || fields[86] != "2" || fields[87] != "2" ||
        !virtual_window::valid_label(fields[29]) ||
        storage >= label_storage.size() || fields[29].size() >= 32u) {
      valid = false;
      continue;
    }
    std::copy(fields[29].begin(), fields[29].end(),
              label_storage[storage].begin());
    labels[packet][rows[packet]] =
        std::string_view{label_storage[storage].data(), fields[29].size()};
    ++storage;
    ++rows[packet];
  }
  valid = valid && packet == std::numeric_limits<std::size_t>::max() &&
          rows == std::array<std::size_t, 3u>{3u, 3u, 3u} &&
          schemas == std::array<bool, 3u>{true, true, true};
  if (!valid) {
    std::fputs("virtual_window_consensus,error,invalid_packet_stream\n",
               stdout);
    return false;
  }
  std::fputs("virtual_window_consensus_columns,q,packet_1,packet_2,packet_3,"
             "published\n",
             stdout);
  for (std::size_t row = 0u; row < 3u; ++row) {
    const bool same =
        labels[0][row] == labels[1][row] && labels[1][row] == labels[2][row];
    std::printf(
        "virtual_window_consensus,%zu,%.*s,%.*s,%.*s,%.*s\n", row + 2u,
        static_cast<int>(labels[0][row].size()), labels[0][row].data(),
        static_cast<int>(labels[1][row].size()), labels[1][row].data(),
        static_cast<int>(labels[2][row].size()), labels[2][row].data(),
        static_cast<int>(same ? labels[0][row].size()
                              : std::string_view{"not_published"}.size()),
        same ? labels[0][row].data() : "not_published");
  }
  return true;
}

} // namespace rund::measure::compute
