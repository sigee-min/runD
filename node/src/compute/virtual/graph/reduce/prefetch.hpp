#pragma once

#include "model.hpp"
#include "wavefront.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>

namespace rund::compute {
class VirtualBacking;
struct Stats;
} // namespace rund::compute

namespace rund::compute::detail {
struct VirtualPipelineState;
struct VirtualSupplyResult;
} // namespace rund::compute::detail

namespace rund::compute::detail::graph_reduce {

class SupplyController;

struct PrefetchLane final {
  residency::PageRun pages{};
  residency::Epoch epoch{};
  VirtualEpochProjection byte_epoch{};
  std::array<residency::PageUse, PipelineLeafCapacity> sources{};
  residency::GraphMaterialization materialization{};
  std::size_t count{};
  std::uint64_t batch{};
  std::uint32_t stage{};
  std::uint32_t resource{};
  std::size_t input_index{};
  std::uint32_t physical_lane{};
  residency::execution::GraphForecast forecast{};
  bool pending{};
  bool speculative{};
  bool device_only{};
};

// Fixed two-lane owner for the Graph product's physical Host Forecast
// lifecycle. The entry point only sequences this controller; projection,
// selection, issue, terminal consumption, and cancellation each have one
// implementation owner under reduce/prefetch/.
class PrefetchController final {
public:
  static constexpr std::size_t LaneCount = 2u;

  PrefetchController(VirtualPipelineState &, std::span<VirtualBacking *const>,
                     const VirtualRunProjection &, Stats &, residency::Pool &,
                     const residency::TiledGraphPlan &, Wavefront &,
                     std::uint64_t batches) noexcept;

  [[nodiscard]] Status schedule(std::uint64_t batch, bool speculative,
                                bool &cleanup_failed) noexcept;
  [[nodiscard]] bool parallel_supported() const noexcept;
  [[nodiscard]] Status refill(Ticket &, bool &cleanup_failed) noexcept;
  [[nodiscard]] Status poll(Ticket &, bool &progressed) noexcept;
  [[nodiscard]] bool has_pending(std::uint64_t batch) const noexcept;
  [[nodiscard]] Status consume(Ticket &, Timeline *hidden_by) noexcept;
  [[nodiscard]] Status supply_stage(Ticket &, const StageScratch &,
                                    const WavefrontCoordinate &,
                                    std::uint32_t resource,
                                    bool &cleanup_failed) noexcept;
  [[nodiscard]] Status replenish(std::uint64_t consumed_batch,
                                 bool &cleanup_failed) noexcept;
  [[nodiscard]] bool cancel() noexcept;
  [[nodiscard]] bool quarantine() noexcept;
  [[nodiscard]] bool quiescent() const noexcept;

private:
  friend class SupplyController;

  [[nodiscard]] Status supply_cpu_stage(Ticket &, PipelineState &,
                                        std::uint32_t stage,
                                        residency::EpochLease,
                                        VirtualSupplyResult &) noexcept;

  [[nodiscard]] PrefetchLane &lane(std::uint64_t batch) noexcept;
  [[nodiscard]] const PrefetchLane &lane(std::uint64_t batch) const noexcept;
  [[nodiscard]] std::size_t lane_index(std::uint64_t batch) const noexcept;
  [[nodiscard]] bool find_input(std::uint32_t resource,
                                std::size_t &index) const noexcept;

  [[nodiscard]] Status
  project(PrefetchLane &, std::uint64_t batch, std::size_t input_index,
          bool speculative, residency::TiledGraphPort &input_port,
          const residency::PoolPhysicalOwner *&input_owner) noexcept;
  [[nodiscard]] Status
  project_stage(PrefetchLane &, const Ticket &, const StageScratch &,
                const WavefrontCoordinate &, std::uint32_t resource,
                const residency::PoolPhysicalOwner *&input_owner) noexcept;
  [[nodiscard]] Status select_missing(PrefetchLane &,
                                      const residency::PoolPhysicalOwner &,
                                      bool speculative) noexcept;
  [[nodiscard]] Status issue(std::size_t lane_index, PrefetchLane &,
                             residency::TiledGraphPort input_port,
                             bool &cleanup_failed) noexcept;
  [[nodiscard]] Status accept(Ticket &, PrefetchLane &,
                              const residency::PrefetchReceipt &) noexcept;
  [[nodiscard]] Status consume(Ticket &, std::uint32_t stage,
                               Timeline *hidden_by) noexcept;
  [[nodiscard]] Status consume_lane(Ticket &, std::size_t lane_index,
                                    Timeline *hidden_by) noexcept;
  [[nodiscard]] bool retire(std::size_t lane_index, PrefetchLane &) noexcept;
  [[nodiscard]] bool settle(PrefetchLane &) noexcept;
  [[nodiscard]] bool workers_quiescent() const noexcept;
  void observe(const residency::PrefetchReceipt &) noexcept;
  [[nodiscard]] Status schedule_input(std::uint64_t batch,
                                      std::size_t input_index, bool speculative,
                                      bool &cleanup_failed) noexcept;

  VirtualPipelineState &state_;
  std::array<VirtualBacking *, residency::execution::GraphPromoteSourceCapacity>
      inputs_{};
  std::size_t input_count_{};
  const VirtualRunProjection &run_;
  Stats &stats_;
  residency::Pool &pool_;
  residency::Authority &authority_;
  const residency::TiledGraphPlan &graph_;
  Wavefront &wavefront_;
  std::uint64_t batches_{};
  std::array<PrefetchLane, LaneCount> lanes_{};
  const bool parallel_{};
};

} // namespace rund::compute::detail::graph_reduce
