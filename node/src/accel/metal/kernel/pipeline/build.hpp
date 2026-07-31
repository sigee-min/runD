#pragma once

#include "state.hpp"
#include "status.hpp"

#include "../../../kernel/recurrence.hpp"

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund::node::accel::detail {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

struct MetalPipelineBuild final {
  std::span<const BackendBatchEntry> templates;
  std::span<const BackendBatchEntry> entries;
  std::span<const std::uint8_t> barriers;
  std::span<const TileTransducer> transducers;
  std::span<const BackendPublish> publications;
  PreparedPipelineStatusLayout &status;
  bool profile_steps{};

  MapRecurrence recurrence{};
  MetalKernelContext context{};
  std::vector<MetalPublish> native_publications;
  std::vector<MetalWindow> native_windows;
  std::shared_ptr<MetalSequence> pipeline;

  std::vector<MetalPipelineStatusBindingRecord> status_bindings;
  std::vector<MetalPipelineStatusSourceMeta> status_sources;
  std::vector<MetalPipelineStatusSourceMeta> occurrence_status_sources;
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
  bool finished{};

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
