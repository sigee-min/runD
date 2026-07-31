#pragma once

#include "buffer.hpp"

#include <array>

namespace node_accel_contract::collective {

[[nodiscard]] inline rund::compute_dsl::ComputeOp BuildFixedLane32Op() {
  std::array<rund::kernel::i32, 8u> input{};
  std::array<rund::kernel::i32, 8u> output{};
  const auto body = rund::compute_dsl::bind(input.size())
                        .fixed<1, 31>()
                        .read<"input">(input.data())
                        .write<"output">(output.data());
  return rund::compute_dsl::def("node-context-collective").on(body).map(
      [](auto i, auto b) {
        auto in = b.template read<"input">();
        auto out = b.template write<"output">();
        out[i] = in[i];
      });
}

}  // namespace node_accel_contract::collective
