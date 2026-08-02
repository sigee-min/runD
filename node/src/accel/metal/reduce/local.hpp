#pragma once

#include "../../reduce/pass.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include <kernel/program/compute/reduce/model.hpp>
#include <memory>
#include <string>

namespace rund::node::accel::detail {

struct MetalReduceEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::ReducePlan plan{};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult output{};
  MetalResidentBufferResult logical_count{};
  MetalRuntimeBuffer partial{};
  MetalRuntimeBuffer status{};
  std::shared_ptr<void> pipeline{};
};

void DestroyMetalReduceEncodeResources(void *raw);
[[nodiscard]] std::string MetalReduceSource(rund::kernel::ReduceOp op,
                                            rund::kernel::u64 block_size,
                                            rund::kernel::ComputeDomain domain);
[[nodiscard]] bool MetalReduceSourceUpperBytes(
    rund::kernel::ReduceOp op, rund::kernel::u64 block_size,
    rund::kernel::ComputeDomain domain, std::uint64_t &upper) noexcept;
[[nodiscard]] bool CompileMetalReducePipeline(
    MetalAdapter &adapter, const rund::kernel::ReducePlan &plan,
    rund::kernel::ComputeDomain domain, std::shared_ptr<void> &out);

} // namespace rund::node::accel::detail
