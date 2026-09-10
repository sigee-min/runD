#pragma once

#include "../reduce.hpp"
#include "abort.hpp"
#include "collective.hpp"
#include "middle.hpp"
#include "output/persist.hpp"
#include "prefetch.hpp"
#include "prefix.hpp"
#include "result.hpp"
#include "stage.hpp"
#include "supply.hpp"
#include "wavefront.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute::detail::graph_reduce {

// Invocation-local wiring for the recurrent graph transaction. The context
// borrows every mutable owner from the coordinator; it owns no graph plan,
// ticket table, Authority state, or callback credential. Keeping this seam to
// references and scalar facts lets phase owners communicate without textual
// implementation inclusion or a second lifecycle authority.
struct GraphExecutionContext final {
  VirtualPipelineState &state;
  std::span<VirtualBacking *const> inputs;
  VirtualBacking *output{};
  const VirtualRunProjection &run;
  Stats &stats;
  VirtualReduction *reduction{};
  residency::Pool &pool;
  residency::Authority &authority;
  const residency::TiledGraphPlan &graph;
  std::span<Ticket> tickets;
  Wavefront &wavefront;
  PrefetchController &prefetch;
  SupplyController &supply;
  StageController &stages;
  CollectiveController &collective;
  MiddleController &middle;
  PersistController &persists;
  AbortController &abort;
  ::rund::node::hash_detail::Fnv &output_hash;
  std::uint64_t identity{};
  std::uint64_t batches{};
  std::uint32_t capacity{};
  std::size_t terminal_stage{};
  bool cpu{};
  bool persisted_output{};
  bool pair_route{};
  bool prefetch_cleanup_failed{};
  bool child_poison{};
};

[[nodiscard]] VirtualGraphResult
finish_failure(GraphExecutionContext &, Status, std::uint64_t page, bool poison,
               Phase, Check, std::uint32_t stage, std::uint64_t batch) noexcept;

[[nodiscard]] Status initialize_ticket(GraphExecutionContext &, Ticket &,
                                       std::uint64_t batch) noexcept;

[[nodiscard]] VirtualGraphResult
prepare_initial(GraphExecutionContext &) noexcept;

[[nodiscard]] VirtualGraphResult execute_batch(GraphExecutionContext &,
                                               std::uint64_t batch) noexcept;

[[nodiscard]] VirtualGraphResult
execute_batches(GraphExecutionContext &) noexcept;

[[nodiscard]] VirtualGraphResult finish_tiled(GraphExecutionContext &) noexcept;

[[nodiscard]] VirtualGraphResult
execute_tiled_graph(VirtualPipelineState &, std::span<VirtualBacking *const>,
                    VirtualBacking *, const VirtualRunProjection &, Stats &,
                    VirtualReduction *) noexcept;

} // namespace rund::compute::detail::graph_reduce
