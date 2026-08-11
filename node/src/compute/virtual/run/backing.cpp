#include "backing.hpp"

#include "../../pipeline/run/clock.hpp"
#include "../../type.hpp"
#include "../backing.hpp"

#include <kernel/core/checked.hpp>
#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/window/model.hpp>
#include <rund/compute/pipeline/shape.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>

namespace rund::compute::detail {
namespace {

template <class T>
void encode_window_identity(const std::uint32_t operation,
                            std::byte *const output) noexcept {
  T value{};
  if (operation == static_cast<std::uint32_t>(kernel::WindowOp::Min)) {
    value = std::numeric_limits<T>::max();
  } else if (operation == static_cast<std::uint32_t>(kernel::WindowOp::Max)) {
    value = std::numeric_limits<T>::lowest();
  }
  std::memcpy(output, &value, sizeof(value));
}

[[nodiscard]] bool window_identity(const Type type,
                                   const std::uint32_t operation,
                                   std::byte *const output) noexcept {
  if (operation != static_cast<std::uint32_t>(kernel::WindowOp::Sum) &&
      operation != static_cast<std::uint32_t>(kernel::WindowOp::Min) &&
      operation != static_cast<std::uint32_t>(kernel::WindowOp::Max)) {
    return false;
  }
  switch (type) {
  case Type::I32:
  case Type::FixedLane32:
    encode_window_identity<std::int32_t>(operation, output);
    return true;
  case Type::U32:
    encode_window_identity<std::uint32_t>(operation, output);
    return true;
  case Type::I64:
  case Type::FixedLane64:
    encode_window_identity<std::int64_t>(operation, output);
    return true;
  case Type::U64:
    encode_window_identity<std::uint64_t>(operation, output);
    return true;
  }
  return false;
}

template <class T>
void encode_reduce_identity(const std::uint32_t operation,
                            std::byte *const output) noexcept {
  T value{};
  if (operation == static_cast<std::uint32_t>(kernel::ReduceOp::Min)) {
    value = std::numeric_limits<T>::max();
  } else if (operation == static_cast<std::uint32_t>(kernel::ReduceOp::Max)) {
    value = std::numeric_limits<T>::lowest();
  }
  std::memcpy(output, &value, sizeof(value));
}

[[nodiscard]] bool reduce_identity(const Type type,
                                   const std::uint32_t operation,
                                   std::byte *const output) noexcept {
  if (operation != static_cast<std::uint32_t>(kernel::ReduceOp::Sum) &&
      operation != static_cast<std::uint32_t>(kernel::ReduceOp::CountNonzero) &&
      operation != static_cast<std::uint32_t>(kernel::ReduceOp::Min) &&
      operation != static_cast<std::uint32_t>(kernel::ReduceOp::Max)) {
    return false;
  }
  switch (type) {
  case Type::I32:
  case Type::FixedLane32:
    encode_reduce_identity<std::int32_t>(operation, output);
    return true;
  case Type::U32:
    encode_reduce_identity<std::uint32_t>(operation, output);
    return true;
  case Type::I64:
  case Type::FixedLane64:
    encode_reduce_identity<std::int64_t>(operation, output);
    return true;
  case Type::U64:
    encode_reduce_identity<std::uint64_t>(operation, output);
    return true;
  }
  return false;
}

} // namespace

using ::rund::detail::counter::Accumulate;

Status
validate_virtual_recovery(const VirtualBacking &input,
                          const VirtualBacking &output,
                          const std::uint64_t required_output_bytes) noexcept {
  const std::uint64_t input_recovery =
      VirtualBackingAccess::recovery_bytes(input);
  const std::uint64_t output_recovery =
      VirtualBackingAccess::recovery_bytes(output);
  return input_recovery == 0u && output_recovery <= required_output_bytes
             ? Status::success()
             : Status::fail(Reason::BufferPoisoned);
}

VirtualSupplyResult
read_virtual_epoch(VirtualBacking &backing, const VirtualEpochProjection &epoch,
                   const VirtualRunProjection &run,
                   const residency::EpochLease lease,
                   const residency::PrefetchReceipt &prefetched,
                   ResidencyStats &stats) noexcept {
  std::array<VirtualRead, PipelineLeafCapacity> ranges{};
  std::array<VirtualInputPageProjection, PipelineLeafCapacity> projected{};
  std::size_t count = 0u;
  VirtualSupplyResult result{};
  if (!prefetched.pages.empty()) {
    Accumulate(stats.backing_io_ns, prefetched.io_ns);
    Accumulate(stats.stall_ns, prefetched.wait_ns);
    Accumulate(stats.overlap_ns, prefetched.io_ns > prefetched.wait_ns
                                     ? prefetched.io_ns - prefetched.wait_ns
                                     : 0u);
    if (prefetched.status) {
      Accumulate(stats.prefetch_count, prefetched.pages.size());
      for (const residency::PrefetchedPage &page : prefetched.pages) {
        Accumulate(stats.backing_read_bytes, page.bytes);
      }
    }
  }
  for (std::size_t index = 0u; index < lease.bindings.size(); ++index) {
    const residency::CacheBinding binding = lease.bindings[index];
    if (!binding.fetch || binding.frame >= run.frame_capacity) {
      continue;
    }
    const std::uint64_t page = epoch.failed_page + index;
    if (!project_virtual_input_page(run, page, projected[index]) ||
        backing.size_bytes() != run.input_capacity_bytes) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
    const VirtualInputPageProjection &input_page = projected[index];
    std::byte *const target =
        run.input_stage + binding.frame * run.input_page_bytes;
    std::memset(target, 0, static_cast<std::size_t>(run.input_page_bytes));
    const auto prefetched_page =
        std::find_if(prefetched.pages.begin(), prefetched.pages.end(),
                     [binding](const residency::PrefetchedPage page) {
                       return page.key == binding.key;
                     });
    if (prefetched_page != prefetched.pages.end()) {
      if (!prefetched.status || prefetched.storage == nullptr ||
          prefetched_page->bytes != input_page.transfer_bytes ||
          prefetched_page->target_offset != input_page.target_offset ||
          prefetched_page->storage_offset >
              std::numeric_limits<std::size_t>::max() - run.input_page_bytes) {
        result.status = prefetched.status
                            ? Status::fail(Reason::PipelineInvalid)
                            : prefetched.status;
        return result;
      }
      std::memcpy(target, prefetched.storage + prefetched_page->storage_offset,
                  static_cast<std::size_t>(run.input_page_bytes));
      ++result.fetched_pages;
      Accumulate(result.backing_bytes, input_page.transfer_bytes);
      continue;
    }
    ranges[count++] = VirtualRead{
        .offset = input_page.logical_offset,
        .bytes = std::span<std::byte>{target + input_page.target_offset,
                                      input_page.transfer_bytes},
    };
  }
  const std::uint64_t started = pipeline_clock();
  const Status status =
      backing.read_batch(std::span<const VirtualRead>{ranges.data(), count});
  Accumulate(stats.backing_io_ns, pipeline_clock() - started);
  if (!status) {
    result.status = status;
    return result;
  }
  Accumulate(result.fetched_pages, count);
  Accumulate(result.late_pages, count);
  for (std::size_t index = 0u; index < count; ++index) {
    Accumulate(result.backing_bytes, ranges[index].bytes.size());
    Accumulate(stats.backing_read_bytes, ranges[index].bytes.size());
  }
  if (run.clamp_window || run.clip_window || run.reduction) {
    const std::size_t element_bytes = static_cast<std::size_t>(
        run.input_page_bytes / run.input_frame_elements);
    std::array<std::byte, sizeof(std::uint64_t)> identity{};
    if ((run.clip_window &&
         !window_identity(run.input_type, run.operation, identity.data())) ||
        (run.reduction &&
         !reduce_identity(run.input_type, run.operation, identity.data()))) {
      result.status = Status::fail(Reason::PrimitiveUnsupported);
      return result;
    }
    for (std::size_t index = 0u; index < lease.bindings.size(); ++index) {
      const residency::CacheBinding binding = lease.bindings[index];
      if (!binding.fetch) {
        continue;
      }
      std::byte *const target =
          run.input_stage + binding.frame * run.input_page_bytes;
      const VirtualInputPageProjection &input_page = projected[index];
      for (std::size_t offset = 0u; offset < input_page.leading_fill_bytes;
           offset += element_bytes) {
        std::memcpy(target + offset,
                    run.clamp_window ? target + input_page.leading_fill_bytes
                                     : identity.data(),
                    element_bytes);
      }
      if (input_page.trailing_fill_offset < run.input_page_bytes) {
        const std::byte *const last =
            target + input_page.trailing_fill_offset - element_bytes;
        for (std::size_t offset = input_page.trailing_fill_offset;
             offset < run.input_page_bytes; offset += element_bytes) {
          std::memcpy(target + offset,
                      run.clamp_window ? last : identity.data(), element_bytes);
        }
      }
    }
  }
  return result;
}

Status schedule_virtual_prefetch(VirtualBacking &backing,
                                 const VirtualEpochProjection &epoch,
                                 const VirtualRunProjection &run,
                                 const residency::Authority &authority,
                                 residency::Prefetcher &prefetcher,
                                 bool &pending) noexcept {
  pending = false;
  if (epoch.page_count == 0u || epoch.page_count > PipelineLeafCapacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::array<residency::CacheKey, PipelineLeafCapacity> keys{};
  std::array<std::uint8_t, PipelineLeafCapacity> resident{};
  std::array<residency::PrefetchRequest, PipelineLeafCapacity> requests{};
  const std::uint64_t backing_id = VirtualBackingAccess::id(backing);
  const std::uint64_t backing_version = VirtualBackingAccess::version(backing);
  for (std::size_t index = 0u;
       index < static_cast<std::size_t>(epoch.page_count); ++index) {
    keys[index] = residency::CacheKey{
        .backing = backing_id,
        .version = backing_version,
        .extent = run.cache_extent,
        .materialization_hi = run.cache_identity_hi,
        .materialization_lo = run.cache_identity_lo,
        .page = epoch.failed_page + index,
    };
  }
  const auto key_span = std::span<const residency::CacheKey>{
      keys.data(), static_cast<std::size_t>(epoch.page_count)};
  if (!authority.probe(key_span, std::span<std::uint8_t>{resident.data(),
                                                         key_span.size()})) {
    return Status::fail(Reason::PipelineInvalid);
  }
  std::size_t count = 0u;
  for (std::size_t index = 0u; index < key_span.size(); ++index) {
    if (resident[index] != 0u) {
      continue;
    }
    VirtualInputPageProjection projected{};
    if (!project_virtual_input_page(run, keys[index].page, projected)) {
      return Status::fail(Reason::PipelineInvalid);
    }
    requests[count++] = residency::PrefetchRequest{
        .key = keys[index],
        .offset = projected.logical_offset,
        .bytes = projected.transfer_bytes,
        .target_offset = projected.target_offset,
    };
  }
  if (count == 0u) {
    return Status::success();
  }
  pending = prefetcher.submit(
      backing,
      std::span<const residency::PrefetchRequest>{requests.data(), count});
  return pending ? Status::success() : Status::fail(Reason::PipelineInvalid);
}

void clear_virtual_recovery(VirtualBacking &backing) noexcept {
  VirtualBackingAccess::clear_recovery(backing);
}

} // namespace rund::compute::detail
