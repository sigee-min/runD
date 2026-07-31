#pragma once

#include <accel/kernel/evidence.hpp>
#include <accel/kernel/run.hpp>

#include <node/accel/context.hpp>

#include "bindings.hpp"

#include <cstddef>

namespace node_accel_contract::scatter::match {

struct RepeatedRun {
  rund::AccelEvidence first{};
  rund::AccelEvidence second{};
};

template <typename T>
[[nodiscard]] RepeatedRun RunTwice(Resources<T> &resources,
                                   const std::size_t tile_count) {
  const auto bindings = Bindings(resources);
  const auto run = [&] {
    return rund::node::accel::RunAccelKernel(
        resources.context, resources.kernel,
        rund::AccelRun{
            .bindings = bindings.data(),
            .binding_count = bindings.size(),
            .tile_count = tile_count,
            .fresh_evidence = true,
        });
  };
  return RepeatedRun{.first = run(), .second = run()};
}

} // namespace node_accel_contract::scatter::match
