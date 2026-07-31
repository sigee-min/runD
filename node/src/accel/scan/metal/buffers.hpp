#pragma once

#include <accel/check.hpp>
#include <kernel/program/compute/scan/model.hpp>

#include <memory>

#include "../../kernel/bindings/scan.hpp"

namespace rund::node::accel::detail {

struct MetalAdapter;

[[nodiscard]] rund::AccelCheck ExecuteMetalScanBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    void *input_buffer, void *output_buffer, bool record_dispatches,
    void *logical_count_buffer, rund::kernel::u32 count_words);
[[nodiscard]] rund::AccelCheck EncodeMetalScanBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    void *input_buffer, void *output_buffer, void *totals_buffer,
    void *status_buffer, void *command_encoder, void *logical_count_buffer,
    rund::kernel::u32 count_words);
[[nodiscard]] rund::AccelCheck EncodePreparedMetalScanBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    void *input_buffer, void *output_buffer, void *totals_buffer,
    void *status_buffer, void *command_encoder,
    const std::shared_ptr<void> &block, const std::shared_ptr<void> &prefix,
    const std::shared_ptr<void> &offset, void *logical_count_buffer,
    rund::kernel::u32 count_words, rund::kernel::u64 input_offset = 0u,
    rund::kernel::u64 output_offset = 0u,
    rund::kernel::u64 logical_count_offset = 0u,
    rund::kernel::u64 totals_offset = 0u);
[[nodiscard]] rund::AccelCheck EncodeMetalScanDeferredOffsetBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *input_buffer, void *output_buffer,
    void *totals_buffer, void *status_buffer, void *command_encoder,
    rund::kernel::u64 input_offset = 0u,
    rund::kernel::u64 output_offset = 0u,
    rund::kernel::u64 totals_offset = 0u);
[[nodiscard]] rund::AccelCheck EncodePreparedOffsetScan(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *input_buffer, void *output_buffer,
    void *totals_buffer, void *status_buffer, void *command_encoder,
    const std::shared_ptr<void> &block, const std::shared_ptr<void> &prefix,
    const std::shared_ptr<void> &offset);
[[nodiscard]] rund::AccelCheck EncodeMetalScanU32FlagBuffers(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, void *flags_buffer,
    std::uint64_t flags_offset, void *output_buffer,
    std::uint64_t output_offset, void *totals_buffer,
    std::uint64_t totals_offset, void *status_buffer, void *command_encoder,
    bool materialize_offsets);

} // namespace rund::node::accel::detail
