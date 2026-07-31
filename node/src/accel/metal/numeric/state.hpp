#pragma once

#include "../buffer/owner.hpp"
#include "../buffer/resident/batch.hpp"
#include "../kernel/ops/model.hpp"
#include "../numeric.hpp"

#include <rund/compute/abi/model.hpp>

#include <array>
#include <cstddef>
#include <memory>

namespace rund::node::accel::detail {

struct NumericParams {
  rund::kernel::u64 op = 0u;
  rund::kernel::u64 layout = 1u;
  rund::kernel::u64 rows = 0u;
  rund::kernel::u64 cols = 0u;
  rund::kernel::u64 inner = 0u;
  rund::kernel::u64 batch_count = 0u;
  rund::kernel::u64 rhs_cols = 0u;
  rund::kernel::u64 value_count = 0u;
  rund::kernel::u64 vector_count = 0u;
  rund::kernel::u32 mode = 0u;
  rund::kernel::u32 aux = 0u;
  rund::kernel::u32 max_iterations = 0u;
};

static_assert(sizeof(NumericParams) == 88u);

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
struct MetalNumericPrepared final {
  MetalAdapter *adapter = nullptr;
  std::array<MetalResidentBufferResult, 5u> buffers{};
  std::size_t buffer_count = 0u;
  std::size_t status_index = 5u;
  rund::kernel::u64 status_count = 0u;
  rund::compute::detail::Primitive status_primitive =
      rund::compute::detail::Primitive::Transform;
  rund::kernel::u64 dispatches = 0u;
  rund::kernel::u64 threads = 0u;
  rund::kernel::u64 groups = 0u;
  rund::kernel::u64 threadgroup = 1u;
  rund::kernel::u64 transform_count = 0u;
  bool grouped = false;
  bool semantic_status = false;
  NumericParams params{};
  MetalRuntimeBuffer twiddle{};
  std::shared_ptr<void> pipeline{};

  ~MetalNumericPrepared();
};
#endif

} // namespace rund::node::accel::detail
