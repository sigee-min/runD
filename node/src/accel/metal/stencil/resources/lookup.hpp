#pragma once

#include <accel/check.hpp>
#include <accel/device.hpp>

#include "../../buffer/resident/batch.hpp"
#include "../../range/local.hpp"

#include <optional>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
[[nodiscard]] inline rund::AccelCheck
LookupMetalStencilResidentBuffers(const rund::AccelDevice &pick,
                                  const RangeBinds &bindings,
                                  std::optional<MetalRangeBinds> &out) {
  out.reset();
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  MetalResidentReq reqs[] = {
      {bindings.input, bindings.input_handle, &input},
      {bindings.output, bindings.output_handle, &output}};
  LookupMetalResidentBatch(pick, reqs, "accel_metal_resident_id_unavailable");
  const char *const reason =
      !input.check.ok ? input.check.reason : output.check.reason;
  std::optional<MetalRangeBinds> range =
      MetalRangeBinds::make(std::move(input), std::move(output));
  if (range.has_value()) {
    out = std::move(range);
    return rund::AccelCheck{true, "ok"};
  }
  return rund::AccelCheck{false, reason};
}
#endif

} // namespace rund::node::accel::detail
