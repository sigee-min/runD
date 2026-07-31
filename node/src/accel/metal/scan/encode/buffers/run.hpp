#pragma once

#include <accel/check.hpp>

#include "../../../../domain.hpp"
#include "offset.hpp"

namespace rund::node::accel::detail {

inline rund::AccelCheck EncodeMetalScanBuffersImpl(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, void *const input_buffer,
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const command_encoder,
    const bool materialize_offsets, const std::shared_ptr<void> *const block,
    const std::shared_ptr<void> *const prefix,
    const std::shared_ptr<void> *const offset, void *const logical_count_buffer,
    const rund::kernel::u32 count_words,
    const rund::kernel::u64 input_offset,
    const rund::kernel::u64 output_offset,
    const rund::kernel::u64 logical_count_offset,
    const rund::kernel::u64 totals_offset) {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  MetalScanEncodeState state{};
  const rund::AccelCheck prepared = PrepareMetalScanEncodeState(
      adapter, desc, plan, input_buffer, output_buffer, totals_buffer,
      status_buffer, command_encoder, state, block, prefix, offset);
  if (!prepared.ok) {
    return prepared;
  }
  const rund::kernel::u32 signed_domain = IsSignedDomain(domain) ? 1u : 0u;
  EncodeMetalScanBlock(desc, input_buffer, output_buffer, totals_buffer,
                       status_buffer, logical_count_buffer, count_words,
                       signed_domain, state, input_offset, output_offset,
                       logical_count_offset, totals_offset);
  if (plan.pass_count == 2u) {
    EncodeMetalScanPrefix(totals_buffer, totals_offset, state);
    if (materialize_offsets) {
      const rund::kernel::u32 inclusive =
          desc.op == rund::kernel::ScanOp::InclusiveSum ? 1u : 0u;
      EncodeMetalScanOffset(output_buffer, totals_buffer, status_buffer,
                            input_buffer, logical_count_buffer, count_words,
                            signed_domain, inclusive, state, input_offset,
                            output_offset, logical_count_offset, totals_offset);
    }
  }
  SetMetalLastError(adapter, "ok");
  return rund::AccelCheck{true, "ok"};
#else
  (void)adapter;
  (void)desc;
  (void)plan;
  (void)domain;
  (void)input_buffer;
  (void)output_buffer;
  (void)totals_buffer;
  (void)status_buffer;
  (void)command_encoder;
  (void)materialize_offsets;
  (void)block;
  (void)prefix;
  (void)offset;
  (void)logical_count_buffer;
  (void)count_words;
  (void)input_offset;
  (void)output_offset;
  (void)logical_count_offset;
  (void)totals_offset;
  return rund::AccelCheck{false, "accel_metal_unavailable"};
#endif
}
rund::AccelCheck EncodeMetalScanBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, void *const input_buffer,
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const command_encoder,
    void *const logical_count_buffer, const rund::kernel::u32 count_words) {
  return EncodeMetalScanBuffersImpl(adapter, desc, plan, domain, input_buffer,
                                    output_buffer, totals_buffer, status_buffer,
                                    command_encoder, true, nullptr, nullptr,
                                    nullptr, logical_count_buffer, count_words);
}

rund::AccelCheck EncodeMetalScanDeferredOffsetBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *const input_buffer,
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const command_encoder,
    const rund::kernel::u64 input_offset,
    const rund::kernel::u64 output_offset,
    const rund::kernel::u64 totals_offset) {
  return EncodeMetalScanBuffersImpl(
      adapter, desc, plan, rund::kernel::ComputeDomain::U32, input_buffer,
      output_buffer, totals_buffer, status_buffer, command_encoder, false,
      nullptr, nullptr, nullptr, nullptr, 0u, input_offset, output_offset, 0u,
      totals_offset);
}

} // namespace rund::node::accel::detail
