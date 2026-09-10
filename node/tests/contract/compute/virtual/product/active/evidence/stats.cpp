#include "local.hpp"

#include <algorithm>
#include <cstdio>

namespace rund_node_test_virtual::product::active::evidence_detail {

[[nodiscard]] bool ExactActiveStats(const rund::compute::Stats &stats,
                                    const rund::compute::Backend backend,
                                    const std::size_t active_count) noexcept {
  const std::uint64_t pages =
      active_count / PageElements +
      static_cast<std::uint64_t>(active_count % PageElements != 0u);
  const std::uint64_t epochs =
      pages / FrameCapacity +
      static_cast<std::uint64_t>(pages % FrameCapacity != 0u);
  const auto &residency = stats.pipeline.residency;
  const RouteKind mode = ClassifyMode(backend, residency, pages);
  const bool native_mode = IsNativeMode(mode);
  const bool device_vsm = mode == RouteKind::DeviceVsm;
  const bool persistent = mode == RouteKind::Persistent;
  const std::uint64_t submits = backend == rund::compute::Backend::Cpu ? 0u
                                : native_mode                          ? 1u
                                                                       : epochs;
  const std::uint64_t supplied =
      residency.page_in_count + residency.cache_hit_count;
  const std::uint64_t upload_submits =
      stats.transfer_submissions.host_to_device;
  const std::uint64_t rolling_upload =
      residency.page_in_count * ElementPageBytes;
  const bool exact_upload_submits =
      rolling_upload == 0u ? upload_submits == 0u
                           : upload_submits != 0u && upload_submits <= epochs &&
                                 upload_submits <= residency.page_in_count;
  const bool no_physical_transfer =
      stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u &&
      upload_submits == 0u && stats.transfer_submissions.device_to_host == 0u;
  const bool exact_rolling_transfer =
      backend == rund::compute::Backend::Vulkan &&
      stats.uploaded_bytes == rolling_upload &&
      stats.downloaded_bytes == pages * ElementPageBytes &&
      exact_upload_submits &&
      stats.transfer_submissions.device_to_host == epochs;
  const bool exact_physical_transfer =
      backend == rund::compute::Backend::Vulkan
          ? no_physical_transfer || exact_rolling_transfer
          : no_physical_transfer;
  const std::uint64_t readable =
      active_count * static_cast<std::uint64_t>(sizeof(std::int32_t));
  return residency.logical_bytes == LogicalBytes * 2u &&
         residency.active_count == active_count &&
         residency.page_bytes == ResidencyPageBytes &&
         residency.page_count == PageCount &&
         residency.frame_capacity == FrameCapacity &&
         residency.resident_frames_peak ==
             std::min(pages, static_cast<std::uint64_t>(FrameCapacity * 2u)) &&
         residency.epoch_count == epochs && supplied == pages &&
         residency.prefetch_count + residency.late_page_count <=
             residency.page_in_count &&
         residency.page_in_bytes ==
             (backend == rund::compute::Backend::Cpu
                  ? residency.backing_read_bytes
              : device_vsm ? readable
              : persistent ? supplied * ElementPageBytes
                           : residency.page_in_count * ElementPageBytes) &&
         residency.backing_read_bytes <= residency.page_in_bytes &&
         residency.page_out_bytes == active_count * sizeof(std::int32_t) &&
         (pages == 0u || native_mode || residency.stall_ns != 0u) &&
         residency.page_out_count == pages &&
         residency.backing_read_bytes <= readable &&
         residency.backing_write_bytes == active_count * sizeof(std::int32_t) &&
         residency.samples_allocation_free(WarmRuns) &&
         residency.failed_page ==
             rund::compute::ResidencyStats::no_failed_page &&
         stats.dispatches == (device_vsm   ? 1u
                              : persistent ? 0u
                                           : pages) &&
         stats.command_submits == submits && exact_physical_transfer &&
         stats.transfer_submissions.device_to_device == 0u &&
         (backend == rund::compute::Backend::Cpu || pages < 2u || native_mode);
}

void ReportActiveFirstFalse(const char *const reason,
                            const std::uint64_t actual,
                            const std::uint64_t expected) noexcept {
  std::fprintf(stderr,
               "virtual active first_false=%s actual=%llu expected=%llu\n",
               reason, static_cast<unsigned long long>(actual),
               static_cast<unsigned long long>(expected));
}

void ReportActiveStatsFirstFalse(const rund::compute::Stats &stats,
                                 const rund::compute::Backend backend,
                                 const std::size_t active_count) noexcept {
  const std::uint64_t pages =
      active_count / PageElements +
      static_cast<std::uint64_t>(active_count % PageElements != 0u);
  const std::uint64_t epochs =
      pages / FrameCapacity +
      static_cast<std::uint64_t>(pages % FrameCapacity != 0u);
  const auto &residency = stats.pipeline.residency;
  const RouteKind mode = ClassifyMode(backend, residency, pages);
  const bool native_mode = IsNativeMode(mode);
  const bool device_vsm = mode == RouteKind::DeviceVsm;
  const bool persistent = mode == RouteKind::Persistent;
  const std::uint64_t submits = backend == rund::compute::Backend::Cpu ? 0u
                                : native_mode                          ? 1u
                                                                       : epochs;
  const std::uint64_t supplied =
      residency.page_in_count + residency.cache_hit_count;
  const std::uint64_t upload_submits =
      stats.transfer_submissions.host_to_device;
  const std::uint64_t rolling_upload =
      residency.page_in_count * ElementPageBytes;
  const bool exact_upload_submits =
      rolling_upload == 0u ? upload_submits == 0u
                           : upload_submits != 0u && upload_submits <= epochs &&
                                 upload_submits <= residency.page_in_count;
  const bool no_physical_transfer =
      stats.uploaded_bytes == 0u && stats.downloaded_bytes == 0u &&
      upload_submits == 0u && stats.transfer_submissions.device_to_host == 0u;
  const bool exact_rolling_transfer =
      backend == rund::compute::Backend::Vulkan &&
      stats.uploaded_bytes == rolling_upload &&
      stats.downloaded_bytes == pages * ElementPageBytes &&
      exact_upload_submits &&
      stats.transfer_submissions.device_to_host == epochs;
  const bool exact_physical_transfer =
      backend == rund::compute::Backend::Vulkan
          ? no_physical_transfer || exact_rolling_transfer
          : no_physical_transfer;
  const std::uint64_t readable =
      active_count * static_cast<std::uint64_t>(sizeof(std::int32_t));
  const auto mode_failure = [&]() noexcept {
    std::fprintf(
        stderr,
        "virtual active first_false=stats.window_mode "
        "actual=handoff:%llu,batch:%llu,queue:%llu,device_vsm:%u,"
        "persistent:%u,stall:%llu expected=device_or_persistent_or_stall\n",
        static_cast<unsigned long long>(residency.window_handoff_count),
        static_cast<unsigned long long>(residency.window_batch_count),
        static_cast<unsigned long long>(residency.window_queue_call_count),
        static_cast<unsigned>(device_vsm), static_cast<unsigned>(persistent),
        static_cast<unsigned long long>(residency.stall_ns));
  };
  if (residency.logical_bytes != LogicalBytes * 2u) {
    ReportActiveFirstFalse("stats.logical_bytes", residency.logical_bytes,
                           LogicalBytes * 2u);
  } else if (residency.active_count != active_count) {
    ReportActiveFirstFalse("stats.active_count", residency.active_count,
                           active_count);
  } else if (residency.page_bytes != ResidencyPageBytes) {
    ReportActiveFirstFalse("stats.page_bytes", residency.page_bytes,
                           ResidencyPageBytes);
  } else if (residency.page_count != PageCount) {
    ReportActiveFirstFalse("stats.page_count", residency.page_count, PageCount);
  } else if (residency.frame_capacity != FrameCapacity) {
    ReportActiveFirstFalse("stats.frame_capacity", residency.frame_capacity,
                           FrameCapacity);
  } else if (residency.resident_frames_peak !=
             std::min(pages, static_cast<std::uint64_t>(FrameCapacity * 2u))) {
    ReportActiveFirstFalse(
        "stats.resident_frames_peak", residency.resident_frames_peak,
        std::min(pages, static_cast<std::uint64_t>(FrameCapacity * 2u)));
  } else if (residency.epoch_count != epochs) {
    ReportActiveFirstFalse("stats.epoch_count", residency.epoch_count, epochs);
  } else if (supplied != pages) {
    ReportActiveFirstFalse("stats.supplied_pages", supplied, pages);
  } else if (residency.prefetch_count + residency.late_page_count >
             residency.page_in_count) {
    ReportActiveFirstFalse("stats.prefetch_plus_late",
                           residency.prefetch_count + residency.late_page_count,
                           residency.page_in_count);
  } else {
    const std::uint64_t expected_page_in_bytes =
        backend == rund::compute::Backend::Cpu ? residency.backing_read_bytes
        : device_vsm                           ? readable
        : persistent                           ? supplied * ElementPageBytes
                     : residency.page_in_count * ElementPageBytes;
    if (residency.page_in_bytes != expected_page_in_bytes) {
      ReportActiveFirstFalse("stats.page_in_bytes", residency.page_in_bytes,
                             expected_page_in_bytes);
    } else if (residency.backing_read_bytes > residency.page_in_bytes) {
      ReportActiveFirstFalse("stats.backing_read_le_page_in",
                             residency.backing_read_bytes,
                             residency.page_in_bytes);
    } else if (residency.page_out_bytes !=
               active_count * sizeof(std::int32_t)) {
      ReportActiveFirstFalse("stats.page_out_bytes", residency.page_out_bytes,
                             active_count * sizeof(std::int32_t));
    } else if (!(pages == 0u || native_mode || residency.stall_ns != 0u)) {
      mode_failure();
    } else if (residency.page_out_count != pages) {
      ReportActiveFirstFalse("stats.page_out_count", residency.page_out_count,
                             pages);
    } else if (residency.backing_read_bytes > readable) {
      ReportActiveFirstFalse("stats.backing_read_le_logical",
                             residency.backing_read_bytes, readable);
    } else if (residency.backing_write_bytes !=
               active_count * sizeof(std::int32_t)) {
      ReportActiveFirstFalse("stats.backing_write_bytes",
                             residency.backing_write_bytes,
                             active_count * sizeof(std::int32_t));
    } else if (!residency.samples_allocation_free(WarmRuns)) {
      ReportActiveFirstFalse("stats.samples_allocation_free",
                             residency.allocation_free_runs, WarmRuns);
    } else if (residency.failed_page !=
               rund::compute::ResidencyStats::no_failed_page) {
      ReportActiveFirstFalse("stats.failed_page", residency.failed_page,
                             rund::compute::ResidencyStats::no_failed_page);
    } else {
      const std::uint64_t expected_dispatches = device_vsm   ? 1u
                                                : persistent ? 0u
                                                             : pages;
      if (stats.dispatches != expected_dispatches) {
        ReportActiveFirstFalse("stats.dispatches", stats.dispatches,
                               expected_dispatches);
      } else if (stats.command_submits != submits) {
        ReportActiveFirstFalse("stats.command_submits", stats.command_submits,
                               submits);
      } else if (!exact_physical_transfer) {
        std::fprintf(stderr,
                     "virtual active first_false=stats.physical_transfer "
                     "actual=uploaded:%llu,downloaded:%llu,h2d:%llu,d2h:%llu "
                     "expected=no_transfer_or_exact_vulkan_rolling\n",
                     static_cast<unsigned long long>(stats.uploaded_bytes),
                     static_cast<unsigned long long>(stats.downloaded_bytes),
                     static_cast<unsigned long long>(upload_submits),
                     static_cast<unsigned long long>(
                         stats.transfer_submissions.device_to_host));
      } else if (stats.transfer_submissions.device_to_device != 0u) {
        ReportActiveFirstFalse("stats.device_to_device",
                               stats.transfer_submissions.device_to_device, 0u);
      } else if (!(backend == rund::compute::Backend::Cpu || pages < 2u ||
                   native_mode)) {
        mode_failure();
      }
    }
  }
}

} // namespace rund_node_test_virtual::product::active::evidence_detail
