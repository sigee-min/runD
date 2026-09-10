#pragma once

#include "../../build.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rund::node::accel::detail::metal_pipeline_program_internal {

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)

// This record is a one-entry borrowed view used only while EncodePrograms is
// composing the existing MetalPipelineBuild. It owns no resource, status, or
// command storage; all mutations remain in that single build object.
struct ProgramEntry final {
  MetalKernelResources *resources{};
  const TileTransducer *transducer{};
  const std::shared_ptr<void> *transducer_resource{};
  const BackendWindow *resident_window{};
  std::uint32_t template_index{};
  PreparedProgramStatusSlice binding_slice{};
  PreparedProgramStatusSlice telemetry_range{};
};

[[nodiscard]] rund::AccelCheck EncodeRecurrence(MetalPipelineBuild &build);

[[nodiscard]] rund::AccelCheck PrepareEntry(MetalPipelineBuild &build,
                                            std::size_t entry_index,
                                            bool &scratch_seen,
                                            ProgramEntry &entry);

[[nodiscard]] rund::AccelCheck
EncodeWindowControl(MetalPipelineBuild &build, std::size_t entry_index,
                    const BackendWindow *resident_window, std::uint32_t stage);

[[nodiscard]] rund::AccelCheck EncodeBody(MetalPipelineBuild &build,
                                          const ProgramEntry &entry,
                                          std::size_t entry_index,
                                          std::size_t program_command_begin);

[[nodiscard]] rund::AccelCheck EncodeStatus(MetalPipelineBuild &build,
                                            const ProgramEntry &entry);

[[nodiscard]] rund::AccelCheck EncodePublications(MetalPipelineBuild &build,
                                                  const ProgramEntry &entry);

#endif

} // namespace rund::node::accel::detail::metal_pipeline_program_internal
