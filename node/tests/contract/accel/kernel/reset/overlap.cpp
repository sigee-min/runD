#include "local.hpp"

#include "src/accel/graph/token/reset.hpp"
#include "src/accel/kernel/reset/overlap.hpp"
#include "src/accel/kernel/reset/proof.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace node_accel_contract::reset_contract {

bool CheckLifetimeOverlap() {
  using rund::node::accel::detail::BoundReset;
  using rund::node::accel::detail::reset::Compatible;
  using rund::node::accel::detail::reset::Project;
  using rund::node::accel::detail::reset::Prove;

  const auto owner = std::make_shared<std::uint32_t>(0u);
  const auto reset = [&](const std::uint64_t offset, const std::uint32_t first,
                         const std::uint32_t last, const bool external = false,
                         const std::uint64_t count = 4u,
                         const std::uint64_t stride = 4u) {
    const rund::kernel::ResidentBufferRef ref{
        .id = 7u,
        .bytes = 64u,
        .offset_bytes = offset,
        .element_bytes = 4u,
        .stride_bytes = stride,
        .count = count,
        .usage = rund::kernel::kResidentUsageWrite,
    };
    const Range proved = Prove(Project(ref, nullptr), ref.bytes);
    return BoundReset::Seal(ref, owner, proved, 0u,
                            rund::node::accel::detail::ExecStep{first},
                            rund::node::accel::detail::ExecStep{last}, external)
        .value();
  };

  const rund::kernel::ResidentBufferRef sealed_source{
      .id = 7u,
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageWrite,
  };
  const Range sealed_range =
      Prove(Project(sealed_source, nullptr), sealed_source.bytes);
  auto mismatched_source = sealed_source;
  mismatched_source.offset_bytes = 4u;
  if (BoundReset::Seal(mismatched_source, owner, sealed_range, 0u,
                       rund::node::accel::detail::ExecStep{0u},
                       rund::node::accel::detail::ExecStep{1u}, false)
          .has_value()) {
    return false;
  }
  auto read_source = sealed_source;
  read_source.usage = rund::kernel::kResidentUsageRead;
  auto unknown_source = sealed_source;
  unknown_source.usage = 0u;
  if (BoundReset::Seal(read_source, owner, sealed_range, 0u,
                       rund::node::accel::detail::ExecStep{0u},
                       rund::node::accel::detail::ExecStep{1u}, false)
          .has_value() ||
      BoundReset::Seal(unknown_source, owner, sealed_range, 0u,
                       rund::node::accel::detail::ExecStep{0u},
                       rund::node::accel::detail::ExecStep{1u}, false)
          .has_value()) {
    return false;
  }

  const std::array disjoint_time{
      reset(0u, 0u, 1u),
      reset(8u, 2u, 3u),
  };
  const std::array touching_time{
      reset(0u, 0u, 1u),
      reset(8u, 1u, 2u),
  };
  const std::array adjacent_memory{
      reset(0u, 0u, 2u),
      reset(16u, 1u, 3u),
  };
  const std::array external_alias{
      reset(0u, 0u, 1u, true),
      reset(8u, 2u, 3u),
  };
  const std::array strided_tail_overlap{
      reset(0u, 0u, 2u, false, 3u, 8u),
      reset(16u, 1u, 3u, false, 1u, 4u),
  };

  return Compatible(disjoint_time) && !Compatible(touching_time) &&
         Compatible(adjacent_memory) && !Compatible(external_alias) &&
         !Compatible(strided_tail_overlap);
}

} // namespace node_accel_contract::reset_contract
