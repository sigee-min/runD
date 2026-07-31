#pragma once

#include "contract/program/compute/lowering/fixed/nonlinear.hpp"

#include <kernel/program/compute/dsl.hpp>
#include <kernel/program/compute/artifact.hpp>
#include <kernel/program/compute/lowering/artifact/admission.hpp>
#include <kernel/program/compute/lowering/entry.hpp>
#include <kernel/program/compute/lowering/fusion/build.hpp>
#include <kernel/program/compute/plan.hpp>

#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace program_compute_contract {

int RunComputeDslIdentityContract();
int RunComputeDslCseContract();
int RunComputeDslOpsContract();
int RunComputeDslRejectContract();
int RunComputeDslEscapeContract();
int RunComputeDslShapeContract();
int RunComputeDslPlanContract();

namespace dsl_support {

using rund::kernel::i32;
using rund::kernel::i64;
using namespace lowering_support;
using namespace nonlinear_support;

template <typename Body, typename Buffer>
concept CanReadBuffer = requires(Body body, Buffer&& buffer) {
  body.template read<"input">(std::forward<Buffer>(buffer));
};

template <typename Body, typename Buffer>
concept CanWriteBuffer = requires(Body body, Buffer&& buffer) {
  body.template write<"output">(std::forward<Buffer>(buffer));
};

[[nodiscard]] inline auto BuildIntegrateBody(const i32 dt) {
  i32 pos[4]{};
  i32 vel[4]{};
  i32 out[4]{};

  return rund::compute_dsl::bind(4u)
      .fixed<1, 31>()
      .param<"dt">(dt)
      .read<"pos">(pos)
      .read<"vel">(vel)
      .write<"out">(out);
}

[[nodiscard]] inline auto BuildIntegrateOp(const i32 dt) {
  auto body = BuildIntegrateBody(dt);

  return rund::compute_dsl::def("integrate").on(body).map([](auto i, auto b) {
    auto dt_value = b.template param<"dt">();
    auto pos = b.template read<"pos">();
    auto vel = b.template read<"vel">();
    auto out = b.template write<"out">();

    out[i] = pos[i] + (vel[i] * dt_value);
  });
}

}  // namespace dsl_support
}  // namespace program_compute_contract
