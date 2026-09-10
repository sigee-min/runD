#pragma once

#include "local.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <vector>

namespace rund_node_test_pipeline::view {

template <typename T>
[[nodiscard]] bool CheckScatterCohortTelemetry(rund::compute::Device &device) {
  using namespace rund::compute;
  for (const std::size_t count : {31u, 32u, 33u, 257u, 1027u}) {
    for (const unsigned pattern : {0u, 1u, 2u}) {
      std::vector<T> values(count);
      std::vector<std::uint32_t> indices(count);
      std::vector<bool> occupied(count + 3u);
      std::uint64_t conflicts = 0u;
      for (std::size_t i = 0u; i < count; ++i) {
        values[i] = static_cast<T>(i * 2654435761u);
        indices[i] = pattern == 0u   ? 1u
                     : pattern == 2u ? static_cast<std::uint32_t>(i)
                     : (i / 32u) % 2u == 0u
                         ? 1u
                         : static_cast<std::uint32_t>(i % 19u);
        conflicts += occupied[indices[i]] ? 1u : 0u;
        occupied[indices[i]] = true;
      }
      auto input = device.upload<T>(values);
      auto targets = device.upload<std::uint32_t>(indices);
      auto output = device.buffer<T>(occupied.size());
      if (!input || !targets || !output) {
        return false;
      }
      for (const auto op : {Reduce::Sum, Reduce::Min, Reduce::Max}) {
        auto program =
            on(device)
                .input<T>(count)
                .template zip_input<std::uint32_t>(count)
                .branch([op, size = occupied.size()](auto source, auto index) {
                  return source.scatter_reduce(index, size, op);
                })
                .compile();
        if (!program) {
          return false;
        }
        auto execution =
            pipeline(device)
                .then(*program, read(*input, *targets), write(*output))
                .prepare();
        if (!execution) {
          return false;
        }
        std::vector<T> expected(
            occupied.size(),
            op == Reduce::Min ? std::numeric_limits<T>::max() : 0u);
        for (std::size_t i = 0u; i < count; ++i) {
          auto &value = expected[indices[i]];
          value = op == Reduce::Sum   ? value + values[i]
                  : op == Reduce::Min ? std::min(value, values[i])
                                      : std::max(value, values[i]);
        }
        std::vector<T> observed(occupied.size());
        for (unsigned repeat = 0u; repeat < 3u; ++repeat) {
          if (!execution->run() ||
              execution->stats().control.conflict_count != conflicts ||
              !execution->read(*output, std::span<T>{observed}) ||
              observed != expected) {
            std::fprintf(
                stderr,
                "scatter cohort success width=%zu count=%zu pattern=%u op=%u "
                "repeat=%u conflicts=%llu expected=%llu\n",
                sizeof(T), count, pattern, static_cast<unsigned>(op), repeat,
                static_cast<unsigned long long>(
                    execution->stats().control.conflict_count),
                static_cast<unsigned long long>(conflicts));
            return false;
          }
        }
        if (op == Reduce::Sum && pattern == 0u) {
          auto invalid_indices = indices;
          invalid_indices[1u] = static_cast<std::uint32_t>(occupied.size());
          invalid_indices.back() = invalid_indices[1u];
          auto invalid_targets = device.upload<std::uint32_t>(invalid_indices);
          auto invalid_output = device.buffer<T>(occupied.size());
          if (!invalid_targets || !invalid_output) {
            return false;
          }
          auto rejected = pipeline(device)
                              .then(*program, read(*input, *invalid_targets),
                                    write(*invalid_output))
                              .prepare();
          if (!rejected) {
            std::fprintf(stderr,
                         "scatter cohort rejection prepare failed: %.*s\n",
                         int(rejected.error().size()), rejected.error().data());
            return false;
          }
          const auto failed = rejected->run();
          if (failed ||
              failed.error() != "compute_scatter_reduce_index_out_of_range" ||
              rejected->stats().control.overflow_ordinal != 1u ||
              rejected->stats().control.conflict_count != 0u) {
            std::fprintf(stderr,
                         "scatter cohort rejection width=%zu count=%zu ok=%u "
                         "reason=%.*s ordinal=%llu conflicts=%llu\n",
                         sizeof(T), count, static_cast<unsigned>(bool(failed)),
                         failed ? 0 : int(failed.error().size()),
                         failed ? "" : failed.error().data(),
                         static_cast<unsigned long long>(
                             rejected->stats().control.overflow_ordinal),
                         static_cast<unsigned long long>(
                             rejected->stats().control.conflict_count));
            return false;
          }
        }
      }
    }
  }
  return true;
}

} // namespace rund_node_test_pipeline::view
