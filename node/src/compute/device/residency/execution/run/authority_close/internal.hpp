#pragma once

#include "../../../registry.hpp"
#include "../../close.hpp"

#include <array>
#include <cstddef>
#include <vector>

namespace rund::compute::detail::residency::execution::authority_close_detail {

struct Validation final {
  AuthorityFailure failure{AuthorityFailure::None};
  ExecutionProgress progress{};
  bool successful{};
};

struct Preparation final {
  close_detail::CloseWork work{};
  std::array<std::size_t, 3u * ExecutionClose::FailureCapacity> failure_nodes{};
  std::size_t failure_node_count{};
  std::size_t cache_dispatch_node{close_detail::CloseNodeCapacity};
  std::size_t success_dispatch_node{close_detail::CloseNodeCapacity};
  std::size_t success_output_node{close_detail::CloseNodeCapacity};
  AuthorityFailure failure{AuthorityFailure::None};
};

[[nodiscard]] Validation validate(const registry_model::ExecutionSlot &slot,
                                  const Plan &plan, const Evidence &evidence,
                                  bool authority_ready) noexcept;

[[nodiscard]] Preparation
prepare(const std::vector<Authority::Frame> &frames,
        const registry_model::ExecutionSlot &slot, const Plan &plan,
        const Evidence &evidence, bool successful) noexcept;

[[nodiscard]] ExecutionClose
apply(std::vector<Authority::Frame> &frames,
      registry_model::ExecutionSlot &slot, const Evidence &evidence,
      const Validation &validation, const Preparation &prepared) noexcept;

} // namespace rund::compute::detail::residency::execution::authority_close_detail
