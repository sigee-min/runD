#pragma once

#include <kernel/program/compute/scan/reference.hpp>
#include <kernel/program/compute/segmented/scan/reference.hpp>

#include "../local.hpp"

#include <array>
#include <cstddef>

namespace node_accel_contract::cpu_context::segmented {

constexpr std::size_t kCount = 8u;

struct Work {
  std::array<rund::kernel::u32, kCount> input{};
  std::array<rund::kernel::u32, kCount> heads{};
  std::array<rund::kernel::u32, kCount> expected_scan{};
  std::array<rund::kernel::i32, kCount> expected_output{};
  std::array<rund::kernel::i32, kCount> map_input{};
  std::array<rund::kernel::i32, kCount> map_output{};
};

[[nodiscard]] inline bool BuildWork(Work &work) {
  work.input = {3u, 1u, 4u, 0u, 2u, 5u, 1u, 6u};
  work.heads = {1u, 0u, 0u, 1u, 0u, 1u, 0u, 0u};
  if (!rund::kernel::ReferenceExclusiveSegmentedScanU32(
           work.input.data(), work.heads.data(), work.expected_scan.data(),
           work.input.size())
           .ok) {
    return false;
  }
  for (std::size_t index = 0u; index < kCount; ++index) {
    work.expected_output[index] =
        static_cast<rund::kernel::i32>(work.expected_scan[index] + 1u);
  }
  return true;
}

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildMapOp(Work &work) {
  const auto body = rund::compute_dsl::bind(kCount)
                        .fixed<1, 31>()
                        .read<"scan">(work.map_input.data())
                        .write<"output">(work.map_output.data());
  return rund::compute_dsl::def("node-cpu-context-segmented-map")
      .on(body)
      .map([](auto i, auto b) {
        auto scan = b.template read<"scan">();
        auto output = b.template write<"output">();
        output[i] = scan[i] + 1;
      });
}

} // namespace node_accel_contract::cpu_context::segmented
