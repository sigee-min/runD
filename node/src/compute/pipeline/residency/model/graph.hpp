#pragma once

#include <array>
#include <span>
#include <vector>

#include <rund/compute/fixed.hpp>
#include <rund/compute/pipeline/shape.hpp>

#include "base.hpp"
#include "stream.hpp"

namespace rund::compute::detail::residency {

enum class ResourcePersistence : std::uint8_t { Backing, Transient };
enum class GraphResourceKind : std::uint8_t {
  ExternalInput,
  Internal,
  ExternalOutput,
};
enum class GraphPageOrigin : std::uint8_t { Begin, End };

struct GraphPageRemap final {
  std::uint32_t source_local{};
  std::uint32_t target_local{};
  GraphPageOrigin source_origin{GraphPageOrigin::Begin};
  [[nodiscard]] constexpr bool
  operator==(const GraphPageRemap &) const noexcept = default;
};

struct TiledGraphResourceInput final {
  std::uint32_t resource{};
  Type type{Type::I32};
  FixedFormat format{};
  std::uint64_t page_bytes{};
  std::uint64_t logical_bytes{};
  GraphResourceKind kind{GraphResourceKind::Internal};
  ResourcePersistence persistence{ResourcePersistence::Backing};
  std::vector<GraphPageRemap> remaps;
  [[nodiscard]] constexpr bool
  operator==(const TiledGraphResourceInput &) const noexcept = default;
};

inline constexpr std::uint32_t NoGraphStage =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint16_t NoGraphControl =
    std::numeric_limits<std::uint16_t>::max();
inline constexpr std::size_t TiledGraphResourceCapacity = 9u;
inline constexpr std::size_t TiledGraphPortCapacity = 16u;
inline constexpr std::size_t GraphPageRemapCapacity =
    TiledGraphPortCapacity * PipelineLeafCapacity;
inline constexpr std::size_t TiledGraphDependencyCapacity =
    TiledGraphPortCapacity;

enum class GraphResourceRole : std::uint8_t {
  Input,
  Intermediate,
  Output,
};

// Planner-sealed physical compatibility and live-interval placement.
struct TiledGraphResource final {
  std::uint32_t resource{};
  std::uint32_t physical_id{};
  Type type{Type::I32};
  FixedFormat format{};
  GraphResourceRole role{GraphResourceRole::Input};
  std::uint32_t color{};
  std::uint32_t first_stage{NoGraphStage};
  std::uint32_t last_stage{NoGraphStage};
  std::uint32_t producer_stage{NoGraphStage};
  std::uint64_t page_bytes{};
  std::uint64_t logical_bytes{};
  GraphResourceKind kind{GraphResourceKind::Internal};
  ResourcePersistence persistence{ResourcePersistence::Backing};
  std::vector<GraphPageRemap> remaps;
  [[nodiscard]] constexpr bool
  operator==(const TiledGraphResource &) const noexcept = default;
};

struct TiledGraphPhysicalClass final {
  std::uint32_t physical_id{};
  GraphResourceRole role{GraphResourceRole::Input};
  Type type{Type::I32};
  FixedFormat format{};
  std::uint32_t color{};
  std::uint64_t page_bytes{};
  [[nodiscard]] constexpr bool
  operator==(const TiledGraphPhysicalClass &) const noexcept = default;
};

enum class StageDomain : std::uint8_t { Tile, TilePartial, Terminal };

struct TiledGraphPort final {
  std::uint32_t resource{};
  Access access{Access::Read};
  std::uint16_t program_port{};
  std::uint32_t next_stage{NoGraphStage};
  [[nodiscard]] bool
  operator==(const TiledGraphPort &) const noexcept = default;
};

struct TiledGraphPortInput final {
  std::uint32_t resource{};
  Access access{Access::Read};
  [[nodiscard]] constexpr bool
  operator==(const TiledGraphPortInput &) const noexcept = default;
};

enum class TiledGraphDependencyPhase : std::uint8_t {
  DispatchComplete,
  ReleaseComplete,
};

struct TiledGraphStageDependency final {
  std::uint32_t stage{};
  TiledGraphDependencyPhase phase{TiledGraphDependencyPhase::DispatchComplete};
  [[nodiscard]] constexpr bool
  operator==(const TiledGraphStageDependency &) const noexcept = default;
  [[nodiscard]] constexpr bool
  operator<(const TiledGraphStageDependency &other) const noexcept {
    return stage < other.stage || (stage == other.stage && phase < other.phase);
  }
};

// Port order is the execution ABI. Dependency vectors are planner-sealed
// resource edges, not authored scheduling hints.
struct TiledGraphStage final {
  std::uint32_t node{};
  StageDomain domain{StageDomain::Tile};
  std::uint16_t active_count_input{NoGraphControl};
  std::vector<TiledGraphPort> ports;
  std::vector<TiledGraphStageDependency> same_batch_predecessors;
  std::vector<TiledGraphStageDependency> prior_batch_predecessors;
  [[nodiscard]] bool
  operator==(const TiledGraphStage &) const noexcept = default;
};

struct TiledGraphStageInput final {
  std::uint32_t node{};
  StageDomain domain{StageDomain::Tile};
  std::vector<TiledGraphPortInput> ports;
  [[nodiscard]] bool
  operator==(const TiledGraphStageInput &) const noexcept = default;
};

// Cold storage remains O(resources + stages + ports), independent of page and
// epoch count. Runtime projections expand only one bounded batch.
struct TiledGraphPlanInput final {
  std::uint64_t page_count{};
  std::uint64_t requested_frames{};
  std::uint64_t max_frames{};
  std::uint64_t prefetch_distance{};
  std::uint64_t graph_fingerprint_hi{};
  std::uint64_t graph_fingerprint_lo{};
  std::vector<TiledGraphResourceInput> resources;
  std::vector<TiledGraphStageInput> stages;
};

struct PlanResult;
class TiledGraphPlan;

struct TiledGraphDependency final {
  std::uint64_t ordinal{};
  std::uint64_t batch{};
  std::uint32_t stage{};
  TiledGraphDependencyPhase phase{TiledGraphDependencyPhase::DispatchComplete};
  [[nodiscard]] constexpr bool
  operator==(const TiledGraphDependency &) const noexcept = default;
};

class TiledGraphInvocation final {
public:
  constexpr TiledGraphInvocation() noexcept = default;
  [[nodiscard]] constexpr bool valid() const noexcept {
    return plan_ != nullptr && page_count_ != 0u && resource_count_ != 0u;
  }
  [[nodiscard]] std::uint64_t page_count() const noexcept;
  [[nodiscard]] std::uint64_t frame_capacity() const noexcept;
  [[nodiscard]] std::uint64_t prefetch_distance() const noexcept;
  [[nodiscard]] std::uint64_t batch_count() const noexcept;
  [[nodiscard]] std::uint64_t epoch_count() const noexcept;
  [[nodiscard]] bool batch(std::uint64_t index, PageRun &run) const noexcept;
  [[nodiscard]] bool supply_stream(StreamPlan &projection) const noexcept;
  [[nodiscard]] bool project(std::uint64_t batch, std::size_t stage,
                             std::span<PageUse> uses,
                             Epoch &epoch) const noexcept;
  [[nodiscard]] bool predecessors(std::uint64_t batch, std::size_t stage,
                                  std::span<TiledGraphDependency> storage,
                                  std::size_t &count) const noexcept;
  [[nodiscard]] bool first_use(std::uint32_t resource, std::uint64_t page,
                               std::uint64_t &use) const noexcept;
  [[nodiscard]] bool page_bytes(std::uint32_t resource, std::uint64_t page,
                                std::uint64_t &bytes) const noexcept;
  [[nodiscard]] bool owned_by(const TiledGraphPlan &) const noexcept;

private:
  friend class TiledGraphPlan;
  const TiledGraphPlan *plan_{};
  std::uint64_t page_count_{};
  std::array<std::uint64_t, TiledGraphResourceCapacity> logical_bytes_{};
  std::size_t resource_count_{};
};

class TiledGraphPlan final {
public:
  TiledGraphPlan() = default;
  [[nodiscard]] constexpr std::uint64_t page_count() const noexcept {
    return page_count_;
  }
  [[nodiscard]] constexpr std::uint64_t frame_capacity() const noexcept {
    return frame_capacity_;
  }
  [[nodiscard]] constexpr std::uint64_t prefetch_distance() const noexcept {
    return prefetch_distance_;
  }
  [[nodiscard]] constexpr std::uint64_t graph_fingerprint_hi() const noexcept {
    return graph_fingerprint_hi_;
  }
  [[nodiscard]] constexpr std::uint64_t graph_fingerprint_lo() const noexcept {
    return graph_fingerprint_lo_;
  }
  [[nodiscard]] constexpr std::uint64_t batch_count() const noexcept {
    return frame_capacity_ == 0u ? 0u
                                 : page_count_ / frame_capacity_ +
                                       static_cast<std::uint64_t>(
                                           page_count_ % frame_capacity_ != 0u);
  }
  [[nodiscard]] std::uint64_t epoch_count() const noexcept;
  [[nodiscard]] constexpr std::size_t stage_count() const noexcept {
    return stages_.size();
  }
  [[nodiscard]] constexpr std::span<const TiledGraphResource>
  resources() const noexcept {
    return resources_;
  }
  [[nodiscard]] constexpr std::span<const TiledGraphStage>
  stages() const noexcept {
    return stages_;
  }
  [[nodiscard]] constexpr std::span<const TiledGraphPhysicalClass>
  physical_classes() const noexcept {
    return physical_classes_;
  }
  [[nodiscard]] const TiledGraphResource *
  resource(std::uint32_t resource) const noexcept;
  [[nodiscard]] bool active(std::uint64_t page_count,
                            std::span<const std::uint64_t> logical_bytes,
                            TiledGraphInvocation &projection) const noexcept;

private:
  friend PlanResult PlanResidency(const TiledGraphPlanInput &input) noexcept;
  friend class TiledGraphInvocation;
  TiledGraphPlan(std::uint64_t page_count, std::uint64_t frame_capacity,
                 std::uint64_t prefetch_distance,
                 std::vector<TiledGraphResource> resources,
                 std::vector<TiledGraphPhysicalClass> physical_classes,
                 std::vector<TiledGraphStage> stages,
                 std::uint64_t graph_fingerprint_hi,
                 std::uint64_t graph_fingerprint_lo) noexcept;

  std::uint64_t page_count_{};
  std::uint64_t frame_capacity_{};
  std::uint64_t prefetch_distance_{};
  std::uint64_t graph_fingerprint_hi_{};
  std::uint64_t graph_fingerprint_lo_{};
  std::vector<TiledGraphResource> resources_;
  std::vector<TiledGraphPhysicalClass> physical_classes_;
  std::vector<TiledGraphStage> stages_;
};

} // namespace rund::compute::detail::residency
