#include "../backing.hpp"

#include "../../../pipeline/run/clock.hpp"

#include <rund/compute/pipeline/shape.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>

namespace rund::compute::detail {

using ::rund::detail::counter::Accumulate;

VirtualSupplyResult read_virtual_epoch(
    VirtualBacking &backing, const VirtualEpochProjection &epoch,
    const VirtualRunProjection &run, const residency::EpochLease lease,
    const residency::PrefetchReceipt &prefetched, ResidencyStats &stats,
    VirtualInputReuseSeed *const reuse) noexcept {
  std::array<VirtualInputPageProjection, PipelineLeafCapacity> projected{};
  std::array<VirtualInputMaterialization, PipelineLeafCapacity> pages{};
  VirtualSupplyResult result{};
  if (lease.bindings.size() > PipelineLeafCapacity) {
    result.status = Status::fail(Reason::PipelineInvalid);
    return result;
  }
  if (!prefetched.pages.empty()) {
    Accumulate(stats.backing_io_ns, prefetched.io_ns);
  }
  for (std::size_t index = 0u; index < lease.bindings.size(); ++index) {
    const residency::CacheBinding binding = lease.bindings[index];
    const std::uint64_t page = epoch.failed_page + index;
    if (!project_virtual_input_page(run, page, projected[index]) ||
        binding.key.page != page ||
        binding.key != virtual_cache_key(run, run.input_backing,
                                         run.input_version, page)) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
    const VirtualInputPageProjection &input_page = projected[index];
    const auto prefetched_page =
        std::find_if(prefetched.pages.begin(), prefetched.pages.end(),
                     [binding](const residency::PrefetchedPage page) {
                       return page.key == binding.key;
                     });
    std::byte *const target = prefetched_page == prefetched.pages.end()
                                  ? virtual_input_frame(run, binding.frame)
                                  : prefetched_page->frame;
    pages[index] = VirtualInputMaterialization{
        .page = page,
        .frame = target,
        .read = binding.fetch && prefetched_page == prefetched.pages.end(),
        .finalize = binding.fetch,
    };
    if (!binding.fetch) {
      continue;
    }
    if (target == nullptr) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
    if (prefetched_page != prefetched.pages.end()) {
      const bool exact_target =
          prefetched.coherent_input
              ? prefetched.token == lease.token &&
                    prefetched_page->physical_frame == binding.frame
              : virtual_host_input_frame(run,
                                         prefetched_page->physical_frame) ==
                    prefetched_page->frame;
      if (!prefetched.status || prefetched_page->frame == nullptr ||
          !exact_target ||
          prefetched_page->bytes != input_page.transfer_bytes ||
          prefetched_page->backing_bytes > prefetched_page->bytes ||
          prefetched_page->target_offset != input_page.target_offset) {
        result.status = prefetched.status
                            ? Status::fail(Reason::PipelineInvalid)
                            : prefetched.status;
        return result;
      }
      if (prefetched_page->fetched) {
        if (prefetched.speculative) {
          Accumulate(stats.prefetch_count, 1u);
        } else {
          Accumulate(result.late_pages, 1u);
        }
      }
      ++result.fetched_pages;
      if (prefetched_page->fetched) {
        Accumulate(result.backing_bytes, prefetched_page->backing_bytes);
        Accumulate(stats.backing_read_bytes, prefetched_page->backing_bytes);
      }
    }
  }
  const bool reads = std::any_of(
      pages.begin(), pages.begin() + lease.bindings.size(),
      [](const VirtualInputMaterialization page) { return page.read; });
  VirtualInputMaterializationResult materialized{};
  if (reads || std::any_of(pages.begin(), pages.begin() + lease.bindings.size(),
                           [](const VirtualInputMaterialization page) {
                             return page.finalize;
                           })) {
    const std::uint64_t started = pipeline_clock();
    materialized =
        materialize_virtual_input(backing, run,
                                  std::span<const VirtualInputMaterialization>{
                                      pages.data(), lease.bindings.size()},
                                  reuse);
    Accumulate(stats.backing_io_ns, pipeline_clock() - started);
    if (!materialized.status) {
      result.status = materialized.status;
      return result;
    }
  }
  Accumulate(result.fetched_pages, materialized.read_pages);
  Accumulate(result.late_pages, materialized.read_pages);
  Accumulate(result.backing_bytes, materialized.backing_bytes);
  Accumulate(stats.backing_read_bytes, materialized.backing_bytes);
  if (reuse != nullptr && !lease.bindings.empty() &&
      pages[lease.bindings.size() - 1u].frame != nullptr) {
    reuse->page = pages[lease.bindings.size() - 1u].page;
    reuse->frame = pages[lease.bindings.size() - 1u].frame;
  }
  return result;
}

} // namespace rund::compute::detail
