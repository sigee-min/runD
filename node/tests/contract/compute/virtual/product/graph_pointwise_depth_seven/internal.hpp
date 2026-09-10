#pragma once

#include "../backing.hpp"
#include "../route.hpp"

#include <rund/compute.hpp>
#include <rund/compute/virtual.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace rund_node_test_virtual::product::graph_pointwise_depth_seven {

inline constexpr std::size_t InputCount = 2u;
inline constexpr std::size_t StageCount = 7u;
inline constexpr std::size_t ResourceCount = InputCount + StageCount;
inline constexpr std::size_t OwnerCount = 2u;
inline constexpr std::size_t EndpointCount = InputCount + 1u;
inline constexpr std::size_t PhysicalCount = OwnerCount + EndpointCount;
inline constexpr std::size_t PortCount = 15u;
inline constexpr std::size_t FrameElements = 16u;
inline constexpr std::size_t PageCount = 5u;
inline constexpr std::size_t TailElements = 7u;
inline constexpr std::size_t FrameCapacity = 2u;
inline constexpr std::size_t BatchCount =
    (PageCount + FrameCapacity - 1u) / FrameCapacity;
inline constexpr std::size_t RunCount = 2u;
// Two input branches need a wider first expression than the three-or-more
// input cases: otherwise the compiler can legally compose the first two Maps
// inside the 1024-node expression envelope and the intended first physical
// stage disappears.
inline constexpr std::size_t InputLeafCount = 64u;
// Two inputs plus seven stage results fill the fixed nine-resource plan and
// fifteen of sixteen planner ports. Adjacent stages cannot be fused, so the
// case authenticates seven distinct Pipeline publication owners.
inline constexpr std::size_t StageLeafCount = 260u;

using Program =
    rund::compute::Program<std::uint64_t(std::uint64_t, std::uint64_t)>;
using Pipeline =
    rund::compute::VirtualPipeline<std::uint64_t(std::uint64_t, std::uint64_t)>;

struct Case final {
  std::vector<std::uint64_t> expected;
  std::array<std::shared_ptr<MemoryVirtualBacking>, InputCount> input_backings;
  std::shared_ptr<MemoryVirtualBacking> output_backing;
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct ResidentCase final {
  std::vector<std::uint64_t> expected;
  std::array<std::shared_ptr<rund::compute::VirtualBacking>, InputCount>
      inputs{};
  std::shared_ptr<rund::compute::VirtualBacking> output;
  Pipeline pipeline;
  std::shared_ptr<rund::compute::detail::VirtualPipelineState> state;
};

struct ResidentObservation final {
  rund::compute::Status status =
      rund::compute::Status::fail(rund::compute::Reason::CompletionInvalid);
  rund::compute::Stats before{};
  rund::compute::Stats after{};
  std::array<std::uint64_t, StageCount> generations_before{};
  std::array<std::uint64_t, StageCount> generations_after{};
  std::uint64_t version_before{};
  std::uint64_t version_after{};
  std::uint64_t output_hash{};
  std::uint64_t proof_digest{};
  std::uint8_t proof_scalar{};
  std::uint8_t proof_domain{};
  std::uint32_t proof_element_bytes{};
  std::uint32_t proof_stage_count{};
  std::uint32_t proof_resource_count{};
  std::uint32_t proof_port_count{};
  std::uint32_t proof_owner_binding_count{};
  std::array<std::uint8_t, StageCount> proof_stage_ports{};
  std::uint32_t wavefront_batch_count{};
  std::uint32_t wavefront_stage_count{};
  std::uint32_t wavefront_frame_capacity{};
  std::uint64_t native_submit_count{};
  std::uint64_t epoch_submit_count{};
  std::uint64_t dispatch_count{};
  std::uint64_t tile_dispatch_count{};
  std::uint64_t final_count{};
  std::uint64_t wavefront_steps{};
  std::uint64_t native_generated_epochs{};
  std::uint64_t native_completed_epochs{};
  std::uint64_t native_proof_hi{};
  std::uint64_t native_proof_lo{};
  std::uint64_t native_generation{};
  std::uint64_t native_nonce{};
  std::uint64_t native_gpu_backing_read_bytes{};
  std::uint64_t native_gpu_backing_write_bytes{};
  std::uint64_t public_handoff_count{};
  std::uint64_t authority_accept_count{};
  std::uint64_t pipeline_terminal_count{};
  std::uint64_t backing_publication_count{};
  std::uint64_t cold_prepare_count{};
  std::uint64_t warm_rearm_count{};
  bool final_received{};
  bool native_may_write{};
  bool output_match{};
  bool graph_resident{};
  bool host_service{};
  bool quarantined{};
  bool staged_output{};
  RouteKind route_kind{RouteKind::Unknown};
  std::uint32_t accepted_owner_mask{};
  std::uint32_t accepted_owner_count{};
};

struct Preparation final {
  std::unique_ptr<Case> value{};
  int reason{};
};

struct ResidentPreparation final {
  std::unique_ptr<ResidentCase> value{};
  int reason{};
};

[[nodiscard]] rund::compute::Result<Program>
build_program(const rund::compute::Device &);
[[nodiscard]] bool validate_program(const Program &) noexcept;
[[nodiscard]] Preparation prepare_case(const rund::compute::Device &);
[[nodiscard]] ResidentPreparation
prepare_resident_case(const rund::compute::Device &);
[[nodiscard]] std::array<std::uint64_t, StageCount>
stage_generations(const Case &) noexcept;
[[nodiscard]] bool
validate_case(Case &, rund::compute::Backend, const rund::compute::Status &,
              std::uint64_t,
              const std::array<std::uint64_t, StageCount> &) noexcept;
[[nodiscard]] bool run_resident_case(
    ResidentCase &, std::array<ResidentObservation, RunCount> &,
    rund::compute::Device &, rund::compute::Backend);
void resident_stage_generations(const ResidentCase &,
                                std::span<std::uint64_t>) noexcept;
[[nodiscard]] bool capture_resident_run(ResidentCase &,
                                         ResidentObservation &) noexcept;
[[nodiscard]] bool validate_resident_case(
    const ResidentCase &, rund::compute::Backend,
    std::span<const ResidentObservation>) noexcept;

} // namespace rund_node_test_virtual::product::graph_pointwise_depth_seven
