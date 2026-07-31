#pragma once

#include <accel/check.hpp>

#include "run.hpp"

namespace rund::node::accel::detail {

rund::AccelCheck EncodePreparedMetalScanBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan,
    const rund::kernel::ComputeDomain domain, void *const input_buffer,
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const command_encoder,
    const std::shared_ptr<void> &block, const std::shared_ptr<void> &prefix,
    const std::shared_ptr<void> &offset, void *const logical_count_buffer,
    const rund::kernel::u32 count_words,
    const rund::kernel::u64 input_offset,
    const rund::kernel::u64 output_offset,
    const rund::kernel::u64 logical_count_offset,
    const rund::kernel::u64 totals_offset) {
  return EncodeMetalScanBuffersImpl(adapter, desc, plan, domain, input_buffer,
                                    output_buffer, totals_buffer, status_buffer,
                                    command_encoder, true, &block, &prefix,
                                    &offset, logical_count_buffer, count_words,
                                    input_offset, output_offset,
                                    logical_count_offset, totals_offset);
}

rund::AccelCheck EncodePreparedOffsetScan(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *const input_buffer,
    void *const output_buffer, void *const totals_buffer,
    void *const status_buffer, void *const command_encoder,
    const std::shared_ptr<void> &block, const std::shared_ptr<void> &prefix,
    const std::shared_ptr<void> &offset) {
  return EncodeMetalScanBuffersImpl(
      adapter, desc, plan, rund::kernel::ComputeDomain::U32, input_buffer,
      output_buffer, totals_buffer, status_buffer, command_encoder, false,
      &block, &prefix, &offset, nullptr, 0u);
}

} // namespace rund::node::accel::detail
