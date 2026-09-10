#include "local.hpp"

#include <cstdio>
#include <string_view>

namespace rund::measure::compute::virtual_crossover::aggregate_detail {
namespace {

namespace vc = ::rund::measure::compute::virtual_crossover;

[[nodiscard]] constexpr bool opposite_winners(const CrossoverLabel left,
                                              const CrossoverLabel right) {
  return (left == CrossoverLabel::Cpu && right == CrossoverLabel::Metal) ||
         (left == CrossoverLabel::Metal && right == CrossoverLabel::Cpu);
}

} // namespace

std::string_view label_name(const CrossoverLabel label) noexcept {
  switch (label) {
  case CrossoverLabel::Cpu:
    return "cpu_observed";
  case CrossoverLabel::Metal:
    return "metal_observed";
  case CrossoverLabel::Indeterminate:
    return "indeterminate";
  case CrossoverLabel::Invalid:
    break;
  }
  return "not_published";
}

void print_consensus(
    const std::array<ConsensusCell, vc::CellCount> &cells) noexcept {
  std::fputs("virtual_crossover_consensus_columns,n,radius,active_ratio_num,"
             "active_ratio_den,packet1_label,packet2_label,packet3_label,"
             "published_label\n",
             stdout);
  for (std::size_t ordinal = 0u; ordinal < cells.size(); ++ordinal) {
    const std::size_t n_index =
        ordinal / (vc::Radii.size() * vc::ActiveRatios.size());
    const std::size_t radius_index =
        ordinal / vc::ActiveRatios.size() % vc::Radii.size();
    const std::size_t ratio_index = ordinal % vc::ActiveRatios.size();
    const auto &cell = cells[ordinal];
    const auto &ratio_value = vc::ActiveRatios[ratio_index];
    std::printf(
        "virtual_crossover_consensus,%zu,%zu,%zu,%zu,%.*s,%.*s,%.*s,%.*s\n",
        vc::LogicalCounts[n_index], vc::Radii[radius_index],
        ratio_value.numerator, ratio_value.denominator,
        static_cast<int>(label_name(cell.labels[0]).size()),
        label_name(cell.labels[0]).data(),
        static_cast<int>(label_name(cell.labels[1]).size()),
        label_name(cell.labels[1]).data(),
        static_cast<int>(label_name(cell.labels[2]).size()),
        label_name(cell.labels[2]).data(),
        static_cast<int>(label_name(cell.published()).size()),
        label_name(cell.published()).data());
  }

  std::fputs("virtual_crossover_bracket_columns,radius,active_ratio_num,"
             "active_ratio_den,lower_n,upper_n,lower_label,upper_label\n",
             stdout);
  std::fputs("virtual_crossover_slice_columns,radius,active_ratio_num,"
             "active_ratio_den,published_cells,bracket_count,"
             "crossover_status\n",
             stdout);
  for (std::size_t radius_index = 0u; radius_index < vc::Radii.size();
       ++radius_index) {
    for (std::size_t ratio_index = 0u; ratio_index < vc::ActiveRatios.size();
         ++ratio_index) {
      std::size_t published = 0u;
      std::size_t brackets = 0u;
      for (std::size_t n_index = 0u; n_index < vc::LogicalCounts.size();
           ++n_index) {
        const std::size_t ordinal =
            (n_index * vc::Radii.size() + radius_index) *
                vc::ActiveRatios.size() +
            ratio_index;
        published += static_cast<std::size_t>(cells[ordinal].published() !=
                                              CrossoverLabel::Invalid);
        if (n_index == 0u) {
          continue;
        }
        const std::size_t previous =
            ((n_index - 1u) * vc::Radii.size() + radius_index) *
                vc::ActiveRatios.size() +
            ratio_index;
        const CrossoverLabel left = cells[previous].published();
        const CrossoverLabel right = cells[ordinal].published();
        if (!opposite_winners(left, right)) {
          continue;
        }
        const auto &ratio_value = vc::ActiveRatios[ratio_index];
        std::printf("virtual_crossover_bracket,%zu,%zu,%zu,%zu,%zu,%.*s,%.*s\n",
                    vc::Radii[radius_index], ratio_value.numerator,
                    ratio_value.denominator, vc::LogicalCounts[n_index - 1u],
                    vc::LogicalCounts[n_index],
                    static_cast<int>(label_name(left).size()),
                    label_name(left).data(),
                    static_cast<int>(label_name(right).size()),
                    label_name(right).data());
        ++brackets;
      }
      const auto &ratio_value = vc::ActiveRatios[ratio_index];
      const std::string_view status = brackets != 0u ? "observed"
                                      : published == vc::LogicalCounts.size()
                                          ? "not_observed_in_grid"
                                          : "not_resolved";
      std::printf("virtual_crossover_slice,%zu,%zu,%zu,%zu,%zu,%.*s\n",
                  vc::Radii[radius_index], ratio_value.numerator,
                  ratio_value.denominator, published, brackets,
                  static_cast<int>(status.size()), status.data());
    }
  }
}

} // namespace rund::measure::compute::virtual_crossover::aggregate_detail

namespace rund::measure::compute {

void PrintVirtualCrossoverColumns() {
  std::fputs(
      "virtual_crossover_columns,evidence_scope,status,n,radius,window_terms,"
      "active_ratio_num,active_ratio_den,active_count,active_ratio_actual,"
      "frame_elements,"
      "active_pages,active_over_device_capacity,semantic_additions,"
      "semantic_additions_per_logical_io_byte,frame_capacity,"
      "host_frame_capacity,graph_hash,plan_hash_hi,plan_hash_lo,"
      "metal_plan_hash_hi,metal_plan_hash_lo,cpu_output_hash,metal_output_hash,"
      "cpu_prepare_us,metal_prepare_us,cpu_first_run_us,metal_first_run_us,"
      "cpu_warm_p25_us,cpu_warm_p50_us,cpu_warm_p75_us,cpu_warm_p95_us,"
      "cpu_warm_mad_us,metal_warm_p25_us,metal_warm_p50_us,"
      "metal_warm_p75_us,metal_warm_p95_us,metal_warm_mad_us,metal_speedup,"
      "robust_classification,"
      "cpu_page_in,cpu_cache_hits,cpu_execution_hit_ratio,cpu_host_supply_hits,"
      "cpu_host_hit_ratio,cpu_evictions,cpu_late,cpu_prefetch,"
      "cpu_backing_read_bytes,cpu_stall_ns,cpu_overlap_ns,cpu_h2d_overlap_ns,"
      "cpu_d2h_overlap_ns,cpu_window_handoffs,cpu_window_batches,"
      "cpu_window_queue_calls,metal_page_in,"
      "metal_cache_hits,metal_execution_hit_ratio,metal_host_supply_hits,"
      "metal_host_hit_ratio,metal_evictions,metal_late,metal_prefetch,"
      "metal_backing_read_bytes,metal_stall_ns,metal_overlap_ns,"
      "metal_h2d_overlap_ns,metal_d2h_overlap_ns,metal_window_handoffs,"
      "metal_window_batches,metal_window_queue_calls,"
      "cpu_uploaded_bytes,cpu_downloaded_bytes,metal_uploaded_bytes,"
      "metal_downloaded_bytes,cpu_command_submits,metal_command_submits,"
      "cpu_submit_wait_ns,metal_submit_wait_ns,cpu_readback_ns,"
      "metal_readback_ns,cpu_kernel_ns,metal_kernel_ns,"
      "sampled_runs,allocation_free_runs\n",
      stdout);
}

} // namespace rund::measure::compute
