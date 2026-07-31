#pragma once

#include "../../../segmented/reduce/model.hpp"

#include "../../buffer/owner.hpp"
#include "../../buffer/resident/batch.hpp"
#include "../../state.hpp"

#include <kernel/program/compute/segmented/reduce/model.hpp>

#include <memory>
#include <string>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

struct MetalSegmentedReduceParams final {
  rund::kernel::u64 count{};
  rund::kernel::u64 block_count{};
  rund::kernel::u64 segments_per_group{};
};

struct MetalSegmentedReducePipelines final {
  std::shared_ptr<void> classify{};
  std::shared_ptr<void> prefix{};
  std::shared_ptr<void> scatter{};
  std::shared_ptr<void> reduce{};
};

struct MetalSegmentedReduceResources final {
  MetalAdapter *adapter = nullptr;
  rund::kernel::SegmentedReducePlan plan{};
  MetalResidentBufferResult input{};
  MetalResidentBufferResult heads{};
  MetalResidentBufferResult output{};
  MetalRuntimeBuffer block_counts{};
  MetalRuntimeBuffer block_offsets{};
  MetalRuntimeBuffer segment_starts{};
  MetalRuntimeBuffer segment_count{};
  MetalRuntimeBuffer dispatch_args{};
  MetalRuntimeBuffer status{};
  rund::kernel::u64 segments_per_group{};
  MetalSegmentedReducePipelines pipelines{};
};

void DestroyMetalSegmentedReduce(void *raw);

[[nodiscard]] std::string
MetalSegmentedReduceSource(rund::kernel::ReduceOp op,
                           rund::kernel::ComputeDomain domain);
[[nodiscard]] std::string
MetalSegmentedReduceKey(const rund::kernel::SegmentedReducePlan &plan,
                        rund::kernel::ComputeDomain domain);
[[nodiscard]] std::string
MetalSegmentedReduceName(const rund::kernel::SegmentedReducePlan &plan,
                         rund::kernel::ComputeDomain domain);

[[nodiscard]] rund::AccelCheck
CompileMetalSegmentedReduce(MetalAdapter &adapter,
                            const rund::kernel::SegmentedReducePlan &plan,
                            rund::kernel::ComputeDomain domain,
                            MetalSegmentedReducePipelines &pipelines);

#endif

} // namespace rund::node::accel::detail
