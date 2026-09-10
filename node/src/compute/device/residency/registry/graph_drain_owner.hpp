#pragma once

#include "../registry.hpp"
#include "../execution/graph_drain.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail::residency {

class Authority;
class ResidencyPlan;

// Stateless direct owner for the callback-return-gated Graph Device-output to
// Host-output migration. Authority remains the sole owner of the physical
// gate, frame table, lease slots, and rollback state; this facet only borrows
// that state and owns the three drain algorithms.
class GraphDrainOwner final {
public:
  explicit GraphDrainOwner(Authority &) noexcept;

  GraphDrainOwner(const GraphDrainOwner &) noexcept = default;
  GraphDrainOwner &operator=(const GraphDrainOwner &) = delete;

  [[nodiscard]] AuthorityResult issue_graph_drain(
      std::shared_ptr<const ResidencyPlan>, const TiledGraphInvocation &,
      std::uint64_t batch, std::size_t stage, std::uint32_t resource,
      std::span<const PageUse>, GraphMaterialization, FrameRegion source,
      FrameRegion target, execution::GraphDrain &) noexcept;
  [[nodiscard]] bool terminal_graph_drain(
      execution::GraphDrain &, Status, execution::TerminalKind, bool may_write,
      std::span<const execution::GraphDrainCompletion>) noexcept;
  [[nodiscard]] bool release_graph_drain(execution::GraphDrain &&) noexcept;

private:
  Authority &authority_;
};

} // namespace rund::compute::detail::residency
