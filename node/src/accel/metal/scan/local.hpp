#pragma once

#include <accel/check.hpp>

#include "../../scan/metal.hpp"
#include "../../scan/shape.hpp"
#include "../adapter.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include "limits.hpp"
#include "pipeline.hpp"
namespace rund::node::accel::detail {

[[nodiscard]] rund::AccelCheck EncodeMetalScanBuffersImpl(
    MetalAdapter &adapter, const rund::kernel::ScanDesc &desc,
    const rund::kernel::ScanPlan &plan, rund::kernel::ComputeDomain domain,
    void *input_buffer, void *output_buffer, void *totals_buffer,
    void *status_buffer, void *command_encoder, bool materialize_offsets,
    const std::shared_ptr<void> *block = nullptr,
    const std::shared_ptr<void> *prefix = nullptr,
    const std::shared_ptr<void> *offset = nullptr,
    void *logical_count_buffer = nullptr, rund::kernel::u32 count_words = 0u,
    rund::kernel::u64 input_offset = 0u, rund::kernel::u64 output_offset = 0u,
    rund::kernel::u64 logical_count_offset = 0u,
    rund::kernel::u64 totals_offset = 0u);

} // namespace rund::node::accel::detail
