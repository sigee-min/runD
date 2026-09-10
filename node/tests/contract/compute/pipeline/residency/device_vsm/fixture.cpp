#include "local.hpp"

namespace rund_node_test_pipeline_residency::device_vsm_test {
namespace {

void Complete(void *const raw, accel::DeviceVsmFinal &&final) noexcept {
  auto *const wait = static_cast<Wait *>(raw);
  if (wait == nullptr) {
    return;
  }
  ++wait->final_count;
  wait->valid = accel::device_vsm_final_valid(wait->request, final);
  wait->final = std::move(final);
}

} // namespace

accel::DeviceVsmCapability Capability() noexcept {
  return accel::DeviceVsmCapability{
      .check = {true, "ok"},
      .retained_bytes = sizeof(FakeState),
      .transient_bytes = 0u,
      .width = 2u,
      .gpu_addressable_backing = true,
      .device_generated_recurrence = true,
      .fixed_native_storage = true,
      .fixed_common_storage = true,
      .one_native_submit = true,
      .host_service_turns_zero = true,
      .host_epoch_callbacks_zero = true,
      .aggregate_terminal_once = true,
      .bounded_page_io = true,
      .physical_ring_storage = true,
  };
}

std::shared_ptr<accel::DeviceVsmProof> Proof(const std::uint64_t pages) {
  constexpr std::uint64_t PayloadBytes = 64u;
  constexpr std::uint64_t TailReduction = 12u;
  const std::uint64_t logical_bytes = pages * PayloadBytes - TailReduction;
  auto proof = std::make_shared<accel::DeviceVsmProof>();
  auto owner = std::make_shared<ProofOwner>();
  proof->identity = accel::DeviceVsmIdentity{
      .hi = 0x4445564943455653ull,
      .lo = pages,
  };
  proof->topology = accel::DeviceVsmTopology::Window;
  proof->semantic_owner = owner;
  owner->artifact.key.api = rund::kernel::ComputeApi::Metal;
  owner->artifact.key.variant =
      rund::kernel::LoweringArtifactVariant::DeviceVsm;
  owner->artifact.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  owner->artifact.ok = true;
  owner->artifact.reason = "ok";
  proof->artifact = &owner->artifact;
  proof->plan.api = rund::kernel::ComputeApi::Metal;
  proof->plan.tile_count = PayloadBytes / sizeof(std::uint32_t);
  proof->plan.input_buffer_count = 1u;
  proof->plan.output_buffer_count = 1u;
  proof->plan.param_bytes = accel::DeviceVsmWindowParameterBytes;
  proof->plan.dispatch_window_tiles = proof->plan.tile_count;
  proof->plan.dispatch_count = 1u;
  proof->plan.ok = true;
  proof->plan.reason = "ok";
  const rund::kernel::ResidentBufferRef input_backing{
      .id = 101u,
      .bytes = logical_bytes,
      .offset_bytes = 0u,
      .element_bytes = sizeof(std::uint32_t),
      .stride_bytes = sizeof(std::uint32_t),
      .count = logical_bytes / sizeof(std::uint32_t),
      .usage = rund::kernel::kResidentUsageRead,
  };
  const auto input_handle = std::make_shared<std::uint8_t>(1u);
  const rund::kernel::ResidentBufferRef output_backing{
      .id = 202u,
      .bytes = logical_bytes,
      .offset_bytes = 0u,
      .element_bytes = sizeof(std::uint32_t),
      .stride_bytes = sizeof(std::uint32_t),
      .count = logical_bytes / sizeof(std::uint32_t),
      .usage = rund::kernel::kResidentUsageWrite,
  };
  const auto output_handle = std::make_shared<std::uint8_t>(2u);
  proof->residents = accel::device_vsm_resident_pair(
      input_backing, input_handle, output_backing, output_handle);
  owner->windows[0u] = rund::kernel::ComputeDispatchWindow{
      .begin_sequence = 0u,
      .tile_count = proof->plan.tile_count,
  };
  proof->windows = owner->windows.data();
  proof->parameters = owner->parameters.data();
  proof->geometry = accel::DeviceVsmPageGeometry{
      .logical_bytes = logical_bytes,
      .payload_bytes = PayloadBytes,
      .frame_bytes = 96u,
      .read_prefix_bytes = 16u,
      .target_offset_bytes = 16u,
      .read_suffix_bytes = 16u,
      .page_count = pages,
      .element_bytes = sizeof(std::uint32_t),
  };
  proof->output_bytes = logical_bytes;
  proof->window = accel::DeviceVsmWindowProof{
      .semantic = rund::kernel::PlanWindow(rund::kernel::WindowDesc{
          .op = rund::kernel::WindowOp::Sum,
          .element = rund::kernel::WindowElement::U32,
          .boundary = rund::kernel::WindowBoundary::Clamp,
          .domain = rund::kernel::ComputeDomain::U32,
          .input_count = 24u,
          .output_count = 24u,
          .window_size = 9u,
          .stride = 1u,
          .pad_left = 4u,
      }),
      .footprint =
          [&]() noexcept {
            accel::DeviceVsmWindowFootprintAuthority authority{};
            static_cast<void>(accel::device_vsm_project_window_footprint(
                proof->geometry, authority));
            return authority;
          }(),
      .range_source_hi = 0x11u,
      .range_source_lo = 0x22u,
      .range_execution_hi = 0x33u,
      .range_execution_lo = 0x44u,
      .workgroup_width = 64u,
      .shared_radius_capacity = 4u,
      .range_path = accel::RangePath::SharedHalo,
      .shared_halo = true,
  };
  proof->window_count = owner->windows.size();
  proof->parameter_bytes = owner->parameters.size();
  proof->width = 2u;
  proof->fixed_common_storage = true;
  return proof;
}

void BuildRequest(Wait &wait, const std::shared_ptr<FakeState> &state,
                  const std::uint64_t pages) {
  wait.request = {};
  wait.submission_control.reset();
  wait.final = {};
  wait.final_count = 0u;
  wait.valid = false;
  wait.request = accel::DeviceVsmRequest{
      .proof = Proof(pages),
      .lowering = state,
      .admission = std::make_shared<const std::uint8_t>(3u),
      .token = 301u,
      .generation = 302u,
      .nonce = 303u,
      .submission_control = &wait.submission_control,
      .final = Complete,
      .user = &wait,
  };
}

} // namespace rund_node_test_pipeline_residency::device_vsm_test
