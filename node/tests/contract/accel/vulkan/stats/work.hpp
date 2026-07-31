#pragma once

#include "local.hpp"

#include <array>

namespace node_accel_contract::vulkan_stats {

struct Work {
  std::array<rund::kernel::u32, 4u> lhs{3u, 5u, 8u, 13u};
  std::array<rund::kernel::u32, 4u> rhs{2u, 4u, 6u, 8u};
  std::array<TileValue, 4u> first{};
  std::array<TileValue, 4u> second{};
  std::array<rund::kernel::u32, 4u> expected{12u, 16u, 21u, 28u};
  std::array<rund::kernel::u32, 1u> params{7u};
  std::array<rund::kernel::u64, 4u> sequence_tiles{0u, 1u, 2u, 3u};
};

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildOp(Work& work) {
  const auto body = rund::compute_dsl::bind(4u)
                        .fixed<1, 31>()
                        .param<"dt">(work.params[0])
                        .read<"lhs">(work.lhs.data())
                        .read<"rhs">(work.rhs.data())
                        .write<"output">(work.first.data());
  return rund::compute_dsl::def("node-vulkan-warm-stats").on(body).map(
      [](auto i, auto b) {
        auto dt = b.template param<"dt">();
        auto lhs = b.template read<"lhs">();
        auto rhs = b.template read<"rhs">();
        auto output = b.template write<"output">();
        output[i] = lhs[i] + rhs[i] + dt;
      });
}

[[nodiscard]] inline std::array<rund::kernel::BufferSpan, 2u>
InputBuffers(Work& work) {
  return {rund::kernel::BufferSpan::contiguous(work.lhs.data(),
                                                      work.lhs.size()),
          rund::kernel::BufferSpan::contiguous(work.rhs.data(),
                                                      work.rhs.size())};
}

}  // namespace node_accel_contract::vulkan_stats
