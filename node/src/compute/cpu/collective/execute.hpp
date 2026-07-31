#pragma once

#include "../state.hpp"

#include <algorithm>
#include <atomic>
#include <limits>

namespace rund::compute::detail {

template <class T>
[[nodiscard]] constexpr CpuCollectiveWide WideValue(const T value) noexcept {
  return static_cast<CpuCollectiveWide>(value);
}

template <class T>
[[nodiscard]] constexpr bool
WideValueFits(const CpuCollectiveWide value) noexcept {
  return value >= WideValue(std::numeric_limits<T>::lowest()) &&
         value <= WideValue(std::numeric_limits<T>::max());
}

template <class T>
[[nodiscard]] kernel::ComputeTileCallbackResult
RunScanTile(CpuCollectiveRun &run, const Scan operation, const CpuPass pass,
            const T *const input, T *const output,
            const kernel::ComputeTile &tile,
            const std::atomic_bool *const cancel = nullptr) noexcept {
  if (input == nullptr || output == nullptr ||
      run.totals.size() <= tile.index || run.prefixes.size() <= tile.index) {
    return {false, "compute_cpu_buffer_invalid"};
  }
  if (cancel != nullptr && cancel->load(std::memory_order_acquire)) {
    return {false, "compute_cancelled"};
  }
  if (pass == CpuPass::ScanLocal) {
    CpuCollectiveWide sum{};
    for (kernel::u32 index = tile.begin; index < tile.end; ++index) {
      sum += WideValue(input[index]);
    }
    run.totals[tile.index] = sum;
    return {};
  }
  CpuCollectiveWide running = run.prefixes[tile.index];
  for (kernel::u32 index = tile.begin; index < tile.end; ++index) {
    if (operation == Scan::InclusiveSum) {
      running += WideValue(input[index]);
    }
    if (!WideValueFits<T>(running)) {
      return {false, "compute_scan_sum_overflow"};
    }
    output[index] = static_cast<T>(running);
    if (operation == Scan::ExclusiveSum) {
      running += WideValue(input[index]);
    }
  }
  if (operation == Scan::ExclusiveSum && !WideValueFits<T>(running)) {
    return {false, "compute_scan_sum_overflow"};
  }
  return {};
}

template <class T>
[[nodiscard]] const char *MergeScanTiles(CpuCollectiveRun &run) noexcept {
  CpuCollectiveWide prefix{};
  for (std::size_t tile = 0u; tile < run.totals.size(); ++tile) {
    run.prefixes[tile] = prefix;
    prefix += run.totals[tile];
  }
  return "pass";
}

template <class T>
[[nodiscard]] kernel::ComputeTileCallbackResult
RunReduceTile(CpuCollectiveRun &run, const kernel::ReduceOp operation,
              const T *const input, const kernel::ComputeTile &tile,
              const std::atomic_bool *const cancel = nullptr) noexcept {
  if (input == nullptr || run.totals.size() <= tile.index) {
    return {false, "compute_cpu_buffer_invalid"};
  }
  if (cancel != nullptr && cancel->load(std::memory_order_acquire)) {
    return {false, "compute_cancelled"};
  }
  CpuCollectiveWide value = operation == kernel::ReduceOp::Min
                                ? WideValue(std::numeric_limits<T>::max())
                            : operation == kernel::ReduceOp::Max
                                ? WideValue(std::numeric_limits<T>::lowest())
                                : CpuCollectiveWide{};
  for (kernel::u32 index = tile.begin; index < tile.end; ++index) {
    if (operation == kernel::ReduceOp::Sum) {
      value += WideValue(input[index]);
    } else if (operation == kernel::ReduceOp::Min) {
      value = std::min(value, WideValue(input[index]));
    } else if (operation == kernel::ReduceOp::Max) {
      value = std::max(value, WideValue(input[index]));
    } else {
      value += input[index] != 0 ? CpuCollectiveWide{1} : CpuCollectiveWide{0};
    }
  }
  run.totals[tile.index] = value;
  return {};
}

template <class T>
[[nodiscard]] const char *MergeReduceTiles(CpuCollectiveRun &run,
                                           const kernel::ReduceOp operation,
                                           T *const output) noexcept {
  if (output == nullptr) {
    return "compute_cpu_buffer_invalid";
  }
  CpuCollectiveWide value = operation == kernel::ReduceOp::Min
                                ? WideValue(std::numeric_limits<T>::max())
                            : operation == kernel::ReduceOp::Max
                                ? WideValue(std::numeric_limits<T>::lowest())
                                : CpuCollectiveWide{};
  for (const CpuCollectiveWide part : run.totals) {
    if (operation == kernel::ReduceOp::Min) {
      value = std::min(value, part);
    } else if (operation == kernel::ReduceOp::Max) {
      value = std::max(value, part);
    } else {
      value += part;
    }
  }
  if (!WideValueFits<T>(value)) {
    return operation == kernel::ReduceOp::Sum ? "compute_reduce_sum_overflow"
                                              : "compute_reduce_count_overflow";
  }
  output[0] = static_cast<T>(value);
  return "pass";
}

} // namespace rund::compute::detail
