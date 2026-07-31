#pragma once

#include "../../sort.hpp"
#include "../adapter.hpp"
#include "../object.hpp"
#include "../pipeline/cache.hpp"
#include "../resident.hpp"
#include "../state.hpp"
#include "source.hpp"
#include <kernel/program/compute/model.hpp>
#include <memory>

namespace rund::node::accel::detail {

struct MetalSortPipelines {
  std::shared_ptr<void> dispatch{};
  std::shared_ptr<void> histogram{};
  std::shared_ptr<void> prefix{};
  std::shared_ptr<void> base{};
  std::shared_ptr<void> scatter{};
};

struct MetalSortEncodeResources {
  MetalAdapter *adapter = nullptr;
  rund::kernel::SortPlan plan{};
  rund::kernel::GraphControl control{};
  MetalResidentBufferResult read_keys{};
  MetalResidentBufferResult read_values{};
  MetalResidentBufferResult write_keys{};
  MetalResidentBufferResult write_values{};
  MetalResidentBufferResult logical_count{};
  MetalRuntimeBuffer temp_keys{};
  MetalRuntimeBuffer temp_values{};
  MetalRuntimeBuffer block_counts{};
  MetalRuntimeBuffer block_offsets{};
  MetalRuntimeBuffer bucket_offsets{};
  MetalRuntimeBuffer dispatch_args{};
  MetalRuntimeBuffer status{};
  MetalSortPipelines pipelines{};
  rund::kernel::u32 block_size = 0u;
  rund::kernel::u32 block_count = 0u;
  bool signed_order = false;
};

struct SortParams {
  rund::kernel::u64 element_count = 0u;
  rund::kernel::u32 block_count = 0u;
  rund::kernel::u32 pass_index = 0u;
  rund::kernel::u32 identity_values = 0u;
  rund::kernel::u32 signed_order = 0u;
  rund::kernel::u32 pass_count = 0u;
  rund::kernel::u32 count_words = 0u;
};

[[nodiscard]] inline SortParams
MetalSortParams(const MetalSortEncodeResources &sort,
                const rund::kernel::u32 pass) noexcept {
  return SortParams{
      sort.plan.element_count,
      sort.block_count,
      pass,
      sort.plan.value == rund::kernel::SortValue::IdentityU32 ? 1u : 0u,
      sort.signed_order ? 1u : 0u,
      sort.plan.radix_pass_count,
      static_cast<rund::kernel::u32>(
          rund::kernel::ComputeCountBytes(sort.plan.count_source) /
          sizeof(rund::kernel::u32))};
}

void DestroyMetalSortEncodeResources(void *raw);
[[nodiscard]] bool
SortBlockShapeOk(const rund::kernel::SortPlan &plan,
                 rund::kernel::u32 &block_size, rund::kernel::u32 &block_count,
                 rund::kernel::u64 &block_table_bytes) noexcept;
[[nodiscard]] bool CompileMetalSortPipelines(MetalAdapter &adapter,
                                             rund::kernel::SortKey key,
                                             rund::kernel::u32 block_size,
                                             MetalSortPipelines &pipelines);

} // namespace rund::node::accel::detail
