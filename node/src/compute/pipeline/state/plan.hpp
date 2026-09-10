#pragma once

#include "base.hpp"
#include "publication.hpp"

#include "../../../accel/kernel/prepared/template/registry/reservation.hpp"
#include "../../cpu/state/arena.hpp"
#include "../../cpu/state/binding.hpp"
#include "../../cpu/state/route.hpp"
#include "../../cpu/state/storage.hpp"
#include "../../device/residency/registry.hpp"
#include "../../job/state.hpp"
#include "../residency/model.hpp"

#include <rund/compute/pipeline/memory.hpp>
#include <rund/compute/resource/plan.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

namespace rund::compute::detail {

namespace residency {
class Pool;
}

// One state-wide immutable control authority. The cold plan owns it until
// admission transfers the same record into PipelineWindow; publications never
// retain count/bounds/final-selector mirrors.
struct PipelineWindowControl final {
  PipelinePublicationViewPlan count{};
  std::uint32_t count_input{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t maximum{};
  std::uint32_t tile{};
  std::uint32_t terminal{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t terminal_output{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t terminal_publication{std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t expected{1u};
  std::uint32_t final{1u};

  [[nodiscard]] constexpr bool
  operator==(const PipelineWindowControl &) const noexcept = default;
};

struct PipelineBuildSnapshot;

struct PipelineResolvedViewPlan final {
  std::uint32_t resource{std::numeric_limits<std::uint32_t>::max()};
  Type declared_type{Type::I32};
  FixedFormat declared_format{};
  ResourceAccess declared_access{ResourceAccess::Read};
  std::uint64_t declared_backing_bytes{};
  std::size_t offset{};
  std::size_t count{};
  std::size_t stride{1u};
  std::size_t element_bytes{};
  std::size_t alignment{};
  std::uint64_t offset_bytes{};
  std::uint64_t stride_bytes{};
  std::uint64_t payload_bytes{};
  std::uint64_t span_bytes{};

  [[nodiscard]] constexpr bool
  operator==(const PipelineResolvedViewPlan &) const noexcept = default;
};

struct PipelineResolvedOutputPlan final {
  PipelineResolvedViewPlan view{};
  std::uint32_t physical{std::numeric_limits<std::uint32_t>::max()};
  bool hidden{};

  [[nodiscard]] constexpr bool
  operator==(const PipelineResolvedOutputPlan &) const noexcept = default;
};

struct PipelineStepResourcePlan final {
  std::vector<PipelineResolvedViewPlan> inputs;
  std::vector<PipelineResolvedOutputPlan> outputs;
  std::vector<std::uint32_t> physical_sources;

  [[nodiscard]] bool
  operator==(const PipelineStepResourcePlan &) const noexcept = default;
};

struct PipelineExternalResourcePlan final {
  std::shared_ptr<BufferState> owner;
};

struct PipelineInternalResourcePlan final {
  std::shared_ptr<BufferState> owner;
  PipelineFill fill{PipelineFill::None};
};

struct PipelineResolvedResourcePlan final {
  std::variant<PipelineExternalResourcePlan, PipelineInternalResourcePlan>
      locator{PipelineExternalResourcePlan{}};
  Type type{Type::I32};
  FixedFormat format{};
  std::uint64_t count{};
  std::uint64_t bytes{};
  std::uint64_t physical_bytes{};
  std::uint32_t first_write{resource::NoNode};
  bool output{};
  bool terminal_publish{};
};

struct PipelineStatePairResourcePlan final {
  PipelineResolvedViewPlan published{};
  PipelineResolvedViewPlan pending{};
  // Prepare-time transactional validation consumes this cold scheduler proof.
  // It is retained only for the pending member instead of mirroring first-use
  // facts on every enduring resource descriptor.
  std::uint32_t pending_first_input{resource::NoNode};
  std::uint32_t pending_first_full_write{resource::NoNode};
};

// One cold workspace route owns both presence and recurrence reuse.  The
// absent sentinel is a complete state: consumers may not infer presence again
// from Program chunks, backend Views, or the global JobArena.
struct PipelineWorkspaceRoute final {
  static constexpr std::size_t absent = std::numeric_limits<std::size_t>::max();

  std::size_t owner{absent};

  [[nodiscard]] constexpr bool present() const noexcept {
    return owner != absent;
  }

  [[nodiscard]] constexpr bool owns(const std::size_t index) const noexcept {
    return owner == index;
  }
};

// Logical residency requirements are frozen into the existing Pipeline plan.
// The planner owns immutable demand and future-use facts;
// PipelineMemoryPlan remains the sole placement projection. The Device
// Registry Authority alone owns mutable mapping, replacement, pin, and dirty
// state across backend-local execution frames plus the Pool's registered Host
// input/output regions. No staging mirror or Pipeline-local page policy exists.
enum class PipelineResidencyStage : std::uint8_t {
  Direct,
  Graph,
};

// A service-free Graph owner is a semantic binding, not the physical first
// stage's ordinary port list.  Keep this private mode explicit so admission
// cannot accidentally treat a stage-local owner as the whole Graph owner.
enum class PipelineResidencySemantic : std::uint8_t {
  None,
  GraphReduction,
  GraphPointwise,
};

struct PipelineResidencySemanticPort final {
  std::uint32_t resource{};
  std::uint16_t program_port{};
  residency::Access access{residency::Access::Read};

  [[nodiscard]] constexpr bool
  operator==(const PipelineResidencySemanticPort &) const noexcept = default;
};

struct PipelineResidencyPort final {
  std::uint32_t resource{};
  std::uint32_t pipeline_resource{};
  std::uint16_t program_port{};
  residency::Access access{residency::Access::Read};
  residency::GraphResourceRole role{residency::GraphResourceRole::Input};
  residency::FrameRegion region{};
  std::uint64_t page_bytes{};
  std::array<residency::GraphPageRemap, PipelineLeafCapacity> remaps{};
  std::size_t remap_count{};
};

struct PipelineResidencyPlan final {
  std::shared_ptr<const residency::ResidencyPlan> pages;
  std::shared_ptr<residency::Pool> pool;
  std::uint64_t logical_bytes{};
  std::uint64_t input_page_bytes{};
  std::uint64_t output_page_bytes{};
  // Exact backend allocation charge of the native reusable transfer arena.
  // Zero on backends that do not retain such an owner.
  std::uint64_t transfer_committed_bytes{};
  std::uint64_t resident_bytes{};
  std::uint32_t first_step{};
  std::uint32_t frame_count{};
  std::uint32_t bank{};
  std::uint32_t graph_stage{residency::NoGraphStage};
  PipelineResidencyStage stage{PipelineResidencyStage::Direct};
  PipelineResidencySemantic semantic{PipelineResidencySemantic::None};
  std::array<PipelineResidencySemanticPort,
             residency::TiledGraphResourceCapacity>
      semantic_ports{};
  std::size_t semantic_port_count{};
};

} // namespace rund::compute::detail
