#pragma once

#include "../registry.hpp"
#include "../execution/graph_persist.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace rund::compute::detail::residency {

// Stateless direct owner for Graph Host-output -> backing persistence and its
// CPU retry lifecycle. Authority remains the sole owner of the gate, frame
// table, fixed Persist slots, and pending CPU reservation; this facet borrows
// those records and owns their issue/terminal/recovery algorithms.
class GraphPersistOwner final {
public:
  explicit GraphPersistOwner(Authority &) noexcept;

  GraphPersistOwner(const GraphPersistOwner &) noexcept = default;
  GraphPersistOwner &operator=(const GraphPersistOwner &) = delete;

  [[nodiscard]] AuthorityResult issue_graph_persist(
      std::shared_ptr<const ResidencyPlan>, const TiledGraphInvocation &,
      std::uint64_t batch, std::size_t stage, std::uint32_t resource,
      std::span<const PageUse>, GraphMaterialization, FrameRegion,
      execution::GraphPersist &) noexcept;
  [[nodiscard]] bool terminal_graph_persist(
      execution::GraphPersist &, Status, execution::TerminalKind, bool may_write,
      std::span<const execution::GraphPersistCompletion>) noexcept;
  [[nodiscard]] bool release_graph_persist(execution::GraphPersist &) noexcept;
  [[nodiscard]] bool release_graph_persist(execution::GraphPersist &&) noexcept;

  [[nodiscard]] AuthorityResult begin_cpu_graph_epoch_retry(
      std::span<const PageUse>, std::span<const GraphPortRequest>,
      std::size_t anchor_port, std::uint64_t epoch, CpuReservationKey,
      const GraphPersistIdentity &) noexcept;
  [[nodiscard]] bool discard_cpu_graph_retry(std::uint64_t domain) noexcept;
  [[nodiscard]] bool
  has_foreign_cpu_graph_retry(std::uint64_t domain) const noexcept;
  [[nodiscard]] bool cpu_graph_retry_active() const noexcept;
  [[nodiscard]] bool has_cpu_graph_retry(std::uint64_t domain) const noexcept;
  [[nodiscard]] bool
  validate_cpu_graph_persist(const execution::GraphPersist &) const noexcept;
  [[nodiscard]] bool recover_cpu_graph_persist(execution::GraphPersist &,
                                               bool unknown) noexcept;

private:
  [[nodiscard]] bool close_graph_persist(execution::GraphPersist &,
                                         bool unknown) noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency
