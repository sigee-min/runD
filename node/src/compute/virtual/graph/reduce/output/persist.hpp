#pragma once

#include "../../../../../hash/fnv.hpp"
#include "../output.hpp"

#include "../../../../device/residency/execution/graph_persist.hpp"
#include "../../../../device/residency/persist.hpp"

#include <array>
#include <span>

namespace rund::compute::detail::graph_reduce::output_persist_detail {

struct PersistIo final {
  Status status{Status::fail(Reason::PipelineInvalid)};
  std::array<residency::execution::GraphPersistCompletion,
             residency::execution::GraphPersistCapacity>
      completions{};
  Interval interval{};
  std::uint64_t bytes{};
  std::size_t page_count{};
  std::size_t completed_pages{};
  bool may_write{};
};

struct Projection final {
  std::array<residency::PersistRequest,
             residency::execution::GraphPersistCapacity>
      requests{};
  std::size_t count{};
  std::uint64_t token{};
};

[[nodiscard]] Status
issue(residency::Authority &,
      const std::shared_ptr<const residency::ResidencyPlan> &,
      const VirtualRunProjection &, const residency::TiledGraphPlan &,
      std::size_t terminal_stage, Ticket &,
      residency::execution::GraphPersist &) noexcept;

[[nodiscard]] PersistIo perform(VirtualBacking &, const VirtualRunProjection &,
                                const residency::execution::GraphPersist &,
                                ::rund::node::hash_detail::Fnv &) noexcept;

[[nodiscard]] bool hash_persisted(const VirtualRunProjection &,
                                  const residency::execution::GraphPersist &,
                                  ::rund::node::hash_detail::Fnv &) noexcept;

[[nodiscard]] bool project(const VirtualRunProjection &, const Ticket &,
                           Projection &) noexcept;

void record(Stats &, std::uint64_t io_ns, std::uint64_t bytes,
            std::size_t completed_pages) noexcept;

[[nodiscard]] bool
cancel_unsubmitted(residency::Authority &,
                   residency::execution::GraphPersist &) noexcept;

} // namespace rund::compute::detail::graph_reduce::output_persist_detail

namespace rund::compute::detail::graph_reduce {

// Owns the two fixed Host-output persistence workers for one Graph invocation.
// Native execution and Drain remain separate; only callback-return retirement
// releases a ticket bank for reuse.
class PersistController final {
public:
  PersistController(residency::Authority &, residency::Pool &,
                    const std::shared_ptr<const residency::ResidencyPlan> &,
                    VirtualBacking *, const VirtualRunProjection &,
                    const residency::TiledGraphPlan &, Stats &, Wavefront &,
                    StageController &, std::size_t terminal_stage,
                    ::rund::node::hash_detail::Fnv &) noexcept;

  [[nodiscard]] Status before_start(std::span<Ticket>, Ticket &,
                                    bool &child_poison) noexcept;
  [[nodiscard]] Status start(Ticket &, Timeline *hidden_by,
                             bool &child_poison) noexcept;
  [[nodiscard]] Status reuse(Ticket &, bool &child_poison) noexcept;
  [[nodiscard]] Status finish(std::span<Ticket>, bool &child_poison) noexcept;
  [[nodiscard]] bool abort(std::span<Ticket>, bool &child_poison) noexcept;

private:
  [[nodiscard]] Status prepare_host_output(Ticket &, Timeline *,
                                           bool &child_poison) noexcept;
  [[nodiscard]] Status persist_cpu(Ticket &, bool &child_poison) noexcept;
  [[nodiscard]] Status submit_async(Ticket &, bool &child_poison) noexcept;
  [[nodiscard]] Status retire(Ticket &, bool &child_poison) noexcept;

  residency::Authority &authority_;
  residency::Pool &pool_;
  const std::shared_ptr<const residency::ResidencyPlan> &owner_;
  VirtualBacking *output_{};
  const VirtualRunProjection &run_;
  const residency::TiledGraphPlan &graph_;
  Stats &stats_;
  Wavefront &wavefront_;
  StageController &stages_;
  std::size_t terminal_stage_{};
  ::rund::node::hash_detail::Fnv &hash_;
  bool parallel_{};
};

} // namespace rund::compute::detail::graph_reduce
