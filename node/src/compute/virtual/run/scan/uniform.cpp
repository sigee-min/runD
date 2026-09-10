#include "../scan.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

namespace rund::compute::detail {
namespace {

template <class T>
[[nodiscard]] T load(const std::byte *const source) noexcept {
  T value{};
  std::memcpy(&value, source, sizeof(value));
  return value;
}

template <class T>
[[nodiscard]] bool add(const T left, const T right, T &result) noexcept {
  if constexpr (std::is_signed_v<T>) {
    const __int128_t sum =
        static_cast<__int128_t>(left) + static_cast<__int128_t>(right);
    if (sum < static_cast<__int128_t>(std::numeric_limits<T>::lowest()) ||
        sum > static_cast<__int128_t>(std::numeric_limits<T>::max())) {
      return false;
    }
    result = static_cast<T>(sum);
    return true;
  } else {
    const __uint128_t sum =
        static_cast<__uint128_t>(left) + static_cast<__uint128_t>(right);
    if (sum > static_cast<__uint128_t>(std::numeric_limits<T>::max())) {
      return false;
    }
    result = static_cast<T>(sum);
    return true;
  }
}

template <class T>
[[nodiscard]] Status prepare_block(VirtualScan &scan, const std::size_t index,
                                   const std::byte *const output_first,
                                   const std::byte *const output_last,
                                   const std::byte *const input_last) noexcept {
  const T prior = load<T>(scan.carry.data());
  T total = load<T>(output_last);
  if (!scan.inclusive && !add(total, load<T>(input_last), total)) {
    return Status::fail(Reason::ScanSumOverflow);
  }
  const T original = scan.inclusive ? load<T>(output_first) : T{};
  T injected = prior;
  if (scan.inclusive && !add(original, prior, injected)) {
    return Status::fail(Reason::ScanSumOverflow);
  }
  T next = prior;
  if (!add(prior, total, next)) {
    return Status::fail(Reason::ScanSumOverflow);
  }
  std::memcpy(scan.original[index].data(), &original, sizeof(original));
  std::memcpy(scan.injected[index].data(), &injected, sizeof(injected));
  std::memcpy(scan.carry.data(), &next, sizeof(next));
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

Status prepare_virtual_scan_uniform(const VirtualEpochProjection &epoch,
                                    const VirtualRunProjection &run,
                                    const residency::EpochLease input_lease,
                                    const residency::EpochLease output_lease,
                                    VirtualScan &scan) noexcept {
  if (!run.scan() || input_lease.bindings.empty() ||
      input_lease.bindings.size() > PipelineLeafCapacity ||
      output_lease.bindings.size() != input_lease.bindings.size() ||
      scan.input_count != input_lease.bindings.size()) {
    return Status::fail(Reason::PipelineInvalid);
  }
  const std::size_t element_bytes = static_cast<std::size_t>(
      run.output_page_bytes / run.input_frame_elements);
  if (element_bytes == 0u || element_bytes > sizeof(std::uint64_t)) {
    return Status::fail(Reason::PipelineInvalid);
  }
  for (std::size_t index = 0u; index < input_lease.bindings.size(); ++index) {
    const residency::CacheBinding binding = input_lease.bindings[index];
    const residency::CacheBinding output_binding = output_lease.bindings[index];
    const std::uint64_t page = epoch.failed_page + index;
    if (scan.input_pages[index] != page ||
        binding.key != virtual_cache_key(run, run.input_backing,
                                         run.input_version, page) ||
        output_binding.key != virtual_output_cache_key(run, run.output_backing,
                                                       run.output_version,
                                                       page)) {
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
    const std::byte *const output_base =
        run.host_output_frame_capacity == 0u
            ? virtual_output_frame(run, output_binding.frame)
            : virtual_host_output_frame(run, output_binding.frame);
    if (output_base == nullptr) {
      return Status::fail(Reason::PipelineInvalid);
    }
    const std::byte *const output_first = output_base + run.output_prefix_bytes;
    const std::byte *const output_last =
        output_first + (logical_count - 1u) * element_bytes;
    const std::byte *const input_last = scan.input_last[index].data();
    const Status prepared = visit(scan.type, [&]<class T>() noexcept {
      return prepare_block<T>(scan, index, output_first, output_last,
                              input_last);
    });
    if (!prepared) {
      scan.failed_page = page;
      return prepared;
    }
  }
  return Status::success();
}

} // namespace rund::compute::detail
