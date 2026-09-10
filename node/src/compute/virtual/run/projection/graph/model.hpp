#pragma once

#include "../../projection.hpp"

#include "../../../../device/residency/pool.hpp"

namespace rund::compute::detail {

struct GraphProjectionInputs final {
  const VirtualBufferState *input{};
  PipelineState *prefix{};
  residency::Pool *pool{};
  const residency::TiledGraphPlan *capacity_plan{};
  const residency::PoolPhysicalOwner *input_owner{};
  const residency::PoolPhysicalOwner *intermediate_owner{};
  const residency::PoolPhysicalOwner *output_owner{};
  std::uint32_t input_resource{};
  std::uint32_t intermediate_resource{};
  std::uint32_t output_resource{};
  std::uint64_t capacity{};
  std::uint64_t input_page_bytes{};
  std::uint64_t intermediate_page_bytes{};
  std::uint64_t control_page_bytes{};
  std::uint64_t output_page_bytes{};
  std::uint64_t input_arena_bytes{};
  std::uint64_t intermediate_arena_bytes{};
  std::uint64_t control_arena_bytes{};
  std::uint64_t output_arena_bytes{};
  std::uint64_t host_input_arena_bytes{};
  std::uint64_t host_output_arena_bytes{};
  bool graph_reduction{};
  bool graph_pointwise{};
};

struct GraphProjectionBanks final {
  std::array<std::byte *, residency::Pool::BankCount> input{};
  std::array<std::byte *, residency::Pool::BankCount> control{};
  std::array<std::byte *, residency::Pool::BankCount> output{};
};

[[nodiscard]] bool
validate_virtual_graph_projection(VirtualPipelineState &state,
                                  GraphProjectionInputs &inputs) noexcept;

[[nodiscard]] bool
bind_virtual_graph_banks(const GraphProjectionInputs &inputs,
                         GraphProjectionBanks &banks) noexcept;

[[nodiscard]] bool materialize_virtual_graph_projection(
    VirtualPipelineState &state, const GraphProjectionInputs &inputs,
    const GraphProjectionBanks &banks, const VirtualActiveProjection &active,
    VirtualRunProjection &projection) noexcept;

} // namespace rund::compute::detail
