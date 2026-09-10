#pragma once

#include "../registry.hpp"
#include "../execution/graph_promote.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::residency {

namespace graph_promote_detail {
struct Group;
} // namespace graph_promote_detail

// Stateless direct owner for the bounded Graph HostReady -> DeviceReady
// promotion protocol. Authority remains the sole owner of the gate, frame
// table, lease slots, and quarantine state; this facet borrows that state and
// owns only the promotion algorithms.
class GraphPromoteOwner final {
public:
  explicit GraphPromoteOwner(Authority &) noexcept;

  GraphPromoteOwner(const GraphPromoteOwner &) noexcept = default;
  GraphPromoteOwner &operator=(const GraphPromoteOwner &) = delete;

  [[nodiscard]] AuthorityResult
  issue_graph_promote(execution::GraphForecast &&,
                      const TiledGraphInvocation &, std::uint64_t batch,
                      std::size_t stage, std::uint32_t resource,
                      std::uint64_t destination_token,
                      execution::GraphPromote &) noexcept;
  [[nodiscard]] AuthorityResult
  issue_graph_promote_group(std::span<execution::GraphForecast>,
                            const TiledGraphInvocation &, std::uint64_t batch,
                            std::size_t stage, std::uint64_t destination_token,
                            execution::GraphPromote &) noexcept;
  [[nodiscard]] AuthorityResult
  issue_graph_promote_group(std::span<execution::GraphReady>,
                            const TiledGraphInvocation &, std::uint64_t batch,
                            std::size_t stage, std::uint64_t destination_token,
                            execution::GraphPromote &) noexcept;
  [[nodiscard]] bool terminal_graph_promote(
      execution::GraphPromote &, Status, execution::TerminalKind,
      bool may_write,
      std::span<const execution::GraphPromoteCompletion>) noexcept;
  [[nodiscard]] bool release_graph_promote(execution::GraphPromote &&) noexcept;

private:
  [[nodiscard]] AuthorityResult validate_graph_promote_group(
      std::span<execution::GraphForecast>, const TiledGraphInvocation &,
      std::uint64_t batch, std::size_t stage, std::uint64_t destination_token,
      graph_promote_detail::Group &) const noexcept;
  [[nodiscard]] AuthorityResult validate_graph_promote_group(
      std::span<execution::GraphReady>, const TiledGraphInvocation &,
      std::uint64_t batch, std::size_t stage, std::uint64_t destination_token,
      graph_promote_detail::Group &) const noexcept;
  [[nodiscard]] AuthorityResult bind_graph_promote_group(
      const TiledGraphInvocation &, std::uint64_t destination_token,
      const graph_promote_detail::Group &, execution::GraphPromote &) noexcept;

  Authority &authority_;
};

} // namespace rund::compute::detail::residency
