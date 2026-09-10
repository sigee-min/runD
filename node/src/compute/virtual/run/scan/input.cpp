#include "../scan.hpp"

#include <algorithm>
#include <cstring>

namespace rund::compute::detail {

Status begin_virtual_scan(const VirtualRunProjection &run,
                          VirtualScan &scan) noexcept {
  scan = VirtualScan{.type = run.input_type, .inclusive = run.inclusive_scan()};
  return run.scan() && run.input_type == run.output_type &&
                 run.input_payload_bytes == run.output_payload_bytes
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

Status
capture_virtual_scan_input(const VirtualEpochProjection &epoch,
                           const VirtualRunProjection &run,
                           const residency::EpochLease input_lease,
                           const residency::PrefetchReceipt *const prefetched,
                           VirtualScan &scan) noexcept {
  scan.input_count = 0u;
  if (!run.scan() || input_lease.bindings.empty() ||
      input_lease.bindings.size() > PipelineLeafCapacity ||
      (run.host_output_frame_capacity != 0u && prefetched == nullptr)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t element_bytes =
      static_cast<std::size_t>(run.input_page_bytes / run.input_frame_elements);
  if (element_bytes == 0u || element_bytes > sizeof(std::uint64_t)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < input_lease.bindings.size(); ++index) {
    const residency::CacheBinding binding = input_lease.bindings[index];
    const std::uint64_t page = epoch.failed_page + index;
    VirtualInputPageProjection input_page{};
    if (binding.key != virtual_cache_key(run, run.input_backing,
                                         run.input_version, page) ||
        !project_virtual_input_page(run, page, input_page)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::uint64_t logical_offset = page * run.output_payload_bytes;
    if (logical_offset >= run.active.output_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::size_t logical_bytes = static_cast<std::size_t>(std::min(
        run.output_payload_bytes, run.active.output_bytes - logical_offset));
    const std::size_t logical_count = logical_bytes / element_bytes;
    if (logical_count == 0u || logical_count * element_bytes != logical_bytes) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::byte *source = nullptr;
    if (prefetched == nullptr) {
      source = virtual_input_frame(run, binding.frame);
    } else {
      const auto found =
          std::find_if(prefetched->pages.begin(), prefetched->pages.end(),
                       [binding](const residency::PrefetchedPage page) {
                         return page.key == binding.key;
                       });
      source = !prefetched->status || found == prefetched->pages.end() ||
                       found->frame == nullptr ||
                       virtual_host_input_frame(run, found->physical_frame) !=
                           found->frame ||
                       found->bytes != input_page.transfer_bytes ||
                       found->target_offset != input_page.target_offset
                   ? nullptr
                   : found->frame;
    }
    const std::size_t input_last_offset =
        static_cast<std::size_t>(run.input_prefix_bytes) +
        (logical_count - 1u) * element_bytes;
    if (source == nullptr || input_last_offset > run.input_page_bytes ||
        element_bytes > run.input_page_bytes - input_last_offset ||
        input_last_offset < input_page.target_offset ||
        input_last_offset - input_page.target_offset >=
            input_page.transfer_bytes ||
        element_bytes > input_page.transfer_bytes -
                            (input_last_offset - input_page.target_offset)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    std::memcpy(scan.input_last[index].data(), source + input_last_offset,
                element_bytes);
    scan.input_pages[index] = page;
  }
  scan.input_count = input_lease.bindings.size();
  return Status::success();
}

} // namespace rund::compute::detail
