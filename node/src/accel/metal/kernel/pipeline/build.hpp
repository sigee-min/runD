#pragma once

#include "aggregate/model.hpp"
#include "state.hpp"
#include "status.hpp"

#include "../../../kernel/backend/pipeline_failure.hpp"
#include "../../../kernel/recurrence.hpp"

#include <rund/compute/pipeline/shape.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// Registry-owned immutable recurrence template. It deliberately retains no
// transformed LoweringArtifact, source string, or copied binding/history
// identity. The common normalized recurrence plan is reconstructed from this
// immutable Program signature when a cold registry lookup needs to compare
// templates; only the prepared native owner survives into the warm path.
struct MetalMapRecurrenceTemplate final {
  MetalKernelTemplateKind kind{MetalKernelTemplateKind::MapRecurrence};
  const BackendRun *signature{};
  bool history{};
  std::shared_ptr<const MetalMapTemplateResources> prepared{};
};

static_assert(std::is_standard_layout_v<MetalMapRecurrenceTemplate>);

struct MetalPipelineBuild final {
  std::span<const BackendBatchEntry> templates;
  std::span<const BackendBatchEntry> entries;
  std::span<const std::uint8_t> barriers;
  std::span<const TileTransducer> transducers;
  std::span<const NestedAggregate> aggregates;
  std::span<const BackendPublish> publications;
  PreparedKernelTemplateRegistry &template_registry;
  PreparedPipelineStatusLayout &status;
  bool profile_steps{};
  PreparedPipelineFailureContext failure_context{};

  MapRecurrence recurrence{};
  MetalKernelContext context{};
  std::array<MetalPublish, rund::compute::detail::PipelineLeafCapacity>
      native_publications{};
  std::size_t native_publication_count{};
  std::vector<MetalWindow> native_windows;
  std::shared_ptr<MetalSequence> pipeline;
  MetalNestedAggregate native_aggregate{};
  bool aggregate_selected{};
  std::uint32_t aggregate_profile_owner{PreparedPipelineNoStep};

  std::vector<MetalPipelineStatusBindingRecord> status_bindings;
  std::vector<MetalPipelineStatusSourceMeta> status_sources;
  std::vector<MetalPipelineStatusEntryMeta> status_entries;
  std::vector<MetalPipelineResetMeta> status_resets;
  std::vector<PreparedProgramStatusSlice> telemetry_steps;
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      binding_slices{};
  std::array<PreparedProgramStatusSlice, PreparedPipelineStepCapacity>
      telemetry_ranges{};
  std::uint32_t raw_status_count{};
  std::uint32_t status_entry_count{};
  std::uint32_t private_raw_count{};

  id<MTLDevice> device = nil;
  std::shared_ptr<void> reset_owner;
  std::shared_ptr<void> import_owner;
  std::shared_ptr<void> reduce_owner;
  std::shared_ptr<void> complete_owner;
  std::shared_ptr<void> telemetry_owner;
  std::shared_ptr<void> publish_owner;
  std::shared_ptr<void> advance_owner;
  id<MTLComputePipelineState> reset = nil;
  id<MTLComputePipelineState> import = nil;
  id<MTLComputePipelineState> reduce = nil;
  id<MTLComputePipelineState> complete = nil;
  id<MTLComputePipelineState> telemetry = nil;
  id<MTLComputePipelineState> publish = nil;
  id<MTLComputePipelineState> advance = nil;
  bool needs_import{};
  bool needs_reset{};
  MetalPipelineStatusParams status_params{};

  MetalCapture captured;
  RUNDMetalPipelineCapture *encoder = nil;
  std::size_t reset_command_count{};
  std::uint32_t import_count{};
  std::uint32_t fold_count{};
  std::uint32_t advance_count{};
  std::uint32_t canonicalize_count{};
  std::uint32_t window_publish_count{};
  bool finished{};

  [[nodiscard]] std::span<MetalPublish> native_publication_rows() noexcept {
    return {native_publications.data(), native_publication_count};
  }

  [[nodiscard]] std::span<const MetalPublish>
  native_publication_rows() const noexcept {
    return {native_publications.data(), native_publication_count};
  }

  [[nodiscard]] rund::AccelCheck Admit();
  [[nodiscard]] rund::AccelCheck Describe();
  [[nodiscard]] rund::AccelCheck Allocate(std::shared_ptr<void> &prepared,
                                          PreparedPipelineMemory &memory);
  [[nodiscard]] rund::AccelCheck Capture();
  [[nodiscard]] rund::AccelCheck Finalize(std::shared_ptr<void> &prepared,
                                          PreparedPipelineMemory &memory);
  [[nodiscard]] rund::AccelCheck
  EncodeTelemetry(PreparedProgramStatusSlice telemetry_slice,
                  PreparedProgramStatusSlice binding_slice,
                  std::uint32_t declared_step);
  [[nodiscard]] rund::AccelCheck EncodePrograms();
};

#endif

} // namespace rund::node::accel::detail
