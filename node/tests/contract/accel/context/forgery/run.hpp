#pragma once

#include <accel/api.hpp>
#include <accel/buffer.hpp>
#include <accel/context/buffer.hpp>
#include <accel/context/value.hpp>
#include <accel/device.hpp>
#include <accel/graph/buffer/ref.hpp>
#include <accel/graph/value.hpp>
#include <accel/graph/node.hpp>

#include <node/accel/pick.hpp>

#include <node/accel/context.hpp>

#include "graph.hpp"

#include <array>
#include <memory>

namespace node_accel_contract {

[[nodiscard]] bool ContextRejectsForgedPickOwnerInSupportPaths() {
  namespace fixture = node_accel_contract::context_forgery;

  rund::AccelDevice pick =
      rund::node::accel::PickAccel(fixture::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    if (!fixture::PickUnavailableReasonIsPrecise(pick, rund::AccelApi::Metal)) {
      return false;
    }
    pick =
        rund::node::accel::PickAccel(fixture::Policy(rund::AccelApi::Vulkan));
  }
  if (!pick.check.ok) {
    return fixture::PickUnavailableReasonIsPrecise(pick,
                                                   rund::AccelApi::Vulkan);
  }

  const rund::AccelContext context = rund::node::accel::OpenAccel(pick);
  if (!context.check.ok) {
    return false;
  }
  const rund::AccelBuffer input = rund::node::accel::CreateAccelBuffer(
      context, fixture::BufferDesc(rund::BufferUsage::ReadOnly));
  const rund::AccelBuffer output = rund::node::accel::CreateAccelBuffer(
      context, fixture::BufferDesc(rund::BufferUsage::WriteOnly));
  if (!input.check.ok || !output.check.ok) {
    return false;
  }

  const rund::node::accel::detail::ContextAdmission admission =
      rund::node::accel::detail::AdmitContextForSupport(context);
  if (!admission.check.ok) {
    return false;
  }

  rund::AccelContext forged = context;
  forged.pick.owner =
      std::shared_ptr<void>(context.pick.owner.get(), [](void *) {});
  if (!fixture::CheckReason(
          rund::node::accel::detail::ValidateAccelBufferForSupport(
              forged, admission, input),
          "accel_context_buffer_invalid")) {
    return false;
  }
  std::array<rund::kernel::i32, 8u> values{};
  if (!fixture::CheckReason(rund::node::accel::UploadAccelBuffer(
                                forged, input, values.data(), sizeof(values)),
                            "accel_context_buffer_invalid")) {
    return false;
  }

  int alias_value{};
  forged = context;
  forged.pick.owner = std::shared_ptr<void>(context.pick.owner, &alias_value);
  if (!fixture::CheckReason(
          rund::node::accel::detail::ValidateAccelBufferForSupport(
              forged, admission, input),
          "accel_context_buffer_invalid")) {
    return false;
  }

  forged = context;
  forged.owner = std::shared_ptr<void>(context.owner, &alias_value);
  if (!fixture::CheckReason(
          rund::node::accel::detail::ValidateAccelBufferForSupport(
              forged, admission, input),
          "accel_context_buffer_invalid")) {
    return false;
  }

  const rund::compute_dsl::ComputeOp op = fixture::BuildFixedLane32Op();
  if (!op.ok()) {
    return false;
  }
  std::array<rund::AccelGraphBufferRef, 2u> refs{};
  std::array<rund::AccelGraphNode, 1u> nodes{};
  const rund::AccelGraph graph =
      fixture::GraphFor(op.ir(), input, output, refs, nodes);
  return fixture::CheckReason(
      rund::node::accel::CompileAccelKernel(forged, graph).check,
      "accel_kernel_graph_invalid");
}

} // namespace node_accel_contract
