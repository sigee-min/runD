#pragma once

#include "work.hpp"

namespace node_accel_contract::cpu_context {

[[nodiscard]] inline rund::compute_dsl::ComputeOp MakeMapHashOp(
    MapHashWork& work) {
  const auto body = rund::compute_dsl::bind(kMapHashCount)
                        .fixed<1, 31>()
                        .read<"input">(work.map_input.data())
                        .write<"output">(work.map_output.data());
  return rund::compute_dsl::def("node-cpu-context-map").on(body).map(
      [](auto i, auto b) {
        auto in = b.template read<"input">();
        auto out = b.template write<"output">();
        out[i] = in[i] + in[i] + 5;
      });
}

}  // namespace node_accel_contract::cpu_context
