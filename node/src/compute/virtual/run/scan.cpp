#include "scan.hpp"

#include "../../pipeline/run/clock.hpp"
#include "../backing.hpp"

#include <rund/counter.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <span>
#include <type_traits>

namespace rund::compute::detail {
namespace {

using ::rund::detail::counter::Accumulate;

template <class T>
[[nodiscard]] T load(const std::byte *const source) noexcept {
  T value{};
  std::memcpy(&value, source, sizeof(value));
  return value;
}

template <class T>
[[nodiscard]] bool add(const T left, const T right, T &result) noexcept {
  if constexpr (std::is_signed_v<T>) {
    const __int128 sum =
        static_cast<__int128>(left) + static_cast<__int128>(right);
    if (sum < static_cast<__int128>(std::numeric_limits<T>::lowest()) ||
        sum > static_cast<__int128>(std::numeric_limits<T>::max())) {
      return false;
    }
    result = static_cast<T>(sum);
    return true;
  } else {
    const unsigned __int128 sum = static_cast<unsigned __int128>(left) +
                                  static_cast<unsigned __int128>(right);
    if (sum > static_cast<unsigned __int128>(std::numeric_limits<T>::max())) {
      return false;
    }
    result = static_cast<T>(sum);
    return true;
  }
}

template <class T>
[[nodiscard]] Status inject(VirtualScan &scan, std::byte *const target,
                            const bool inclusive) noexcept {
  const T carry = load<T>(scan.carry.data());
  T value = carry;
  if (inclusive && !add(load<T>(target), carry, value)) {
    return Status::fail(Reason::ScanSumOverflow);
  }
  std::memcpy(target, &value, sizeof(value));
  return Status::success();
}

template <class T>
[[nodiscard]] Status
advance(VirtualScan &scan, const std::byte *const output_last,
        const std::byte *const input_last, const bool inclusive) noexcept {
  T carry = load<T>(output_last);
  if (!inclusive && !add(carry, load<T>(input_last), carry)) {
    return Status::fail(Reason::ScanSumOverflow);
  }
  std::memcpy(scan.carry.data(), &carry, sizeof(carry));
  return Status::success();
}

template <class Fn>
[[nodiscard]] Status visit(const Type type, Fn &&function) noexcept {
  switch (type) {
  case Type::I32:
  case Type::FixedLane32:
    return function.template operator()<std::int32_t>();
  case Type::U32:
    return function.template operator()<std::uint32_t>();
  case Type::I64:
  case Type::FixedLane64:
    return function.template operator()<std::int64_t>();
  case Type::U64:
    return function.template operator()<std::uint64_t>();
  }
  return Status::fail(Reason::PrimitiveUnsupported);
}

} // namespace

Status begin_virtual_scan(const VirtualRunProjection &run,
                          VirtualScan &scan) noexcept {
  scan = VirtualScan{.type = run.input_type, .inclusive = run.inclusive_scan};
  return run.scan && run.input_type == run.output_type &&
                 run.input_payload_bytes == run.output_payload_bytes &&
                 run.frame_capacity == 1u
             ? Status::success()
             : Status::fail(Reason::PipelineInvalid);
}

Status prepare_virtual_scan_page(const VirtualRunProjection &run,
                                 const residency::EpochLease lease,
                                 VirtualScan &scan) noexcept {
  if (!run.scan || lease.bindings.size() != 1u ||
      lease.bindings.front().frame >= run.frame_capacity) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::CacheBinding binding = lease.bindings.front();
  const std::size_t offset =
      static_cast<std::size_t>(binding.frame * run.input_page_bytes);
  // Carry is invocation-local. Copy the canonical host image into the
  // disposable output-staging half before injection so a warm cache hit never
  // mutates the cache key's materialized input page.
  std::memcpy(run.output_stage + offset, run.input_stage + offset,
              static_cast<std::size_t>(run.input_page_bytes));
  std::byte *const target =
      run.output_stage + binding.frame * run.input_page_bytes;
  return visit(scan.type, [&]<class T>() noexcept {
    return inject<T>(scan, target, scan.inclusive);
  });
}

Status complete_virtual_scan_page(VirtualBacking &input,
                                  const VirtualEpochProjection &epoch,
                                  const VirtualRunProjection &run,
                                  const residency::EpochLease lease,
                                  VirtualScan &scan,
                                  ResidencyStats &stats) noexcept {
  if (!run.scan || lease.bindings.size() != 1u ||
      lease.bindings.front().frame >= run.frame_capacity ||
      epoch.logical_output_bytes == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const residency::CacheBinding binding = lease.bindings.front();
  const std::size_t element_bytes = static_cast<std::size_t>(
      run.output_page_bytes / run.input_frame_elements);
  const std::size_t logical_count = epoch.logical_output_bytes / element_bytes;
  if (element_bytes == 0u || logical_count == 0u) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::byte *const output_last =
      run.output_stage + binding.frame * run.output_page_bytes +
      run.output_prefix_bytes + (logical_count - 1u) * element_bytes;
  std::array<std::byte, sizeof(std::uint64_t)> last_storage{};
  const std::byte *input_last =
      run.input_stage + binding.frame * run.input_page_bytes +
      run.input_prefix_bytes + (logical_count - 1u) * element_bytes;
  if (!scan.inclusive && !binding.fetch) {
    const std::uint64_t offset =
        epoch.input_offset + (logical_count - 1u) * element_bytes;
    const std::uint64_t started = pipeline_clock();
    const Status read = input.read(
        offset, std::span<std::byte>{last_storage.data(), element_bytes});
    Accumulate(stats.backing_io_ns, pipeline_clock() - started);
    if (!read) {
      return read;
    }
    Accumulate(stats.backing_read_bytes, element_bytes);
    input_last = last_storage.data();
  }
  return visit(scan.type, [&]<class T>() noexcept {
    return advance<T>(scan, output_last, input_last, scan.inclusive);
  });
}

} // namespace rund::compute::detail
