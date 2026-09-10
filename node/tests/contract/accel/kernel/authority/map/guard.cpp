#include "src/accel/kernel/backend/manifest.hpp"
#include "src/accel/kernel/backend/run.hpp"
#include "src/accel/kernel/backend/source/storage.hpp"
#include "src/accel/kernel/backend/template/identity.hpp"
#include "src/accel/kernel/backend/template/source.hpp"
#include "src/accel/kernel/prepared/template/registry.hpp"
#include "src/accel/kernel/step/map/stride.hpp"

#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
#include "src/accel/metal/kernel/manifest.hpp"
#include "src/accel/metal/kernel/pipeline/source.hpp"
#include "src/accel/metal/pipeline/guard.hpp"
#include "src/accel/metal/runtime/api.hpp"
#include "src/accel/metal/runtime/map/api.hpp"
#include "src/accel/metal/runtime/map/control.hpp"
#include "src/accel/metal/runtime/map/resources.hpp"
#include "src/accel/metal/runtime/map/source/upper.hpp"
#endif

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "../map.hpp"

namespace node_accel_contract {

[[nodiscard]] bool MetalMapCheckSourceHasOneGuardAuthority() {
#if defined(__APPLE__) && defined(RUND_NODE_HAVE_METAL_SDK)
  using namespace rund::node::accel::detail;
  MetalMapTemplateResources prepared{};
  prepared.plan.scalar = rund::kernel::ComputeScalar::Lane32;
  prepared.plan.domain = rund::kernel::ComputeDomain::U32;
  prepared.plan.op_hash_hi = 0x6d61702e63686563ull;
  prepared.plan.op_hash_lo = 0x6b2e736f75726365ull;
  prepared.checks.push_back(MetalMapCheck{.binding = 0u, .limit = 123u});
  const rund::kernel::ResidentBufferRef input{
      .bytes = 64u,
      .element_bytes = 4u,
      .stride_bytes = 8u,
      .count = 8u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  rund::kernel::BindingSet bindings{};
  bindings.resident_inputs = rund::kernel::ResidentBindingRange{
      .refs = &input,
      .storage_count = 1u,
      .count = 1u,
  };

  const KernelPreparationScope scope{KernelPreparationMode::PipelinePrivate};
  rund::kernel::LoweringArtifact artifact =
      MetalMapCheckArtifact(prepared, bindings);
  std::uint64_t raw_upper = 0u;
  std::uint64_t guarded_upper = 0u;
  const bool upper_ok =
      MetalMapCheckSourceUpperBytes(1u, DecimalDigitCount(input.stride_bytes),
                                    DecimalDigitCount(123u), raw_upper);
  const bool guarded_ok = upper_ok && PipelinePrivateMetalSourceUpperBytes(
                                          raw_upper, 1u, true, guarded_upper);
  if (!artifact.ok || !guarded_ok || artifact.source_text.size() != raw_upper ||
      artifact.source_text_upper_bytes != raw_upper ||
      guarded_upper <= raw_upper) {
    return false;
  }
  const std::size_t check_capacity = artifact.source_text.capacity();
  std::string guarded = PipelinePrivateMetalSource(
      std::move(artifact.source_text), guarded_upper);
  std::uint64_t guarded_storage_upper = 0u;
  if (guarded.size() != guarded_upper || guarded.capacity() != check_capacity ||
      guarded.find("buffer(30)") == std::string::npos ||
      !backend_source_recipe::string_external_storage_upper_bytes(
          guarded_upper, guarded_storage_upper) ||
      !backend_source_recipe::string_external_storage_within(
          guarded, guarded_storage_upper)) {
    return false;
  }

  std::string over_capacity = guarded;
  over_capacity.reserve(static_cast<std::size_t>(guarded_storage_upper + 64u));
  if (!PipelinePrivateMetalSource(std::move(over_capacity), guarded_upper)
           .empty()) {
    return false;
  }

  std::uint64_t control_guarded_upper = 0u;
  if (!PipelinePrivateMetalSourceUpperBytes(MetalMapControlSourceText().size(),
                                            1u, true, control_guarded_upper)) {
    return false;
  }
  std::string control_source = MetalMapControlSource();
  const std::size_t control_capacity = control_source.capacity();
  control_source = PipelinePrivateMetalSource(std::move(control_source),
                                              control_guarded_upper);
  if (control_source.size() != control_guarded_upper ||
      control_source.capacity() != control_capacity) {
    return false;
  }

  rund::kernel::LoweringArtifact controlled_input{};
  controlled_input.key.api = rund::kernel::ComputeApi::Metal;
  controlled_input.key.op_hash_hi = 0x1111111111111111ull;
  controlled_input.key.op_hash_lo = 0x2222222222222222ull;
  controlled_input.kind = rund::kernel::LoweringArtifactKind::MetalSource;
  constexpr std::string_view CanonicalControlledSource =
      "// artifact_variant=canonical\n"
      "kernel void "
      "rund_compute_map_1111111111111111_2222222222222222(\n"
      "    uint gid [[thread_position_in_grid]]) {\n"
      "}\n";
  controlled_input.source_text_upper_bytes = CanonicalControlledSource.size();
  controlled_input.ok = true;
  controlled_input.reason = "ok";
  const rund::kernel::ComputePlan controlled_plan{
      .api = rund::kernel::ComputeApi::Metal,
      .input_buffer_count = 1u,
      .output_buffer_count = 1u,
  };
  std::uint64_t controlled_upper = 0u;
  std::uint64_t controlled_guarded_upper = 0u;
  if (!MetalControlledMapSourceUpperBytes(controlled_plan,
                                          CanonicalControlledSource.size(),
                                          controlled_upper) ||
      !PipelinePrivateMetalSourceUpperBytes(controlled_upper, 1u, true,
                                            controlled_guarded_upper)) {
    return false;
  }
  const auto canonical_recipe =
      [source = CanonicalControlledSource]<typename Sink>(Sink &sink) noexcept(
          noexcept(sink.append(std::string_view{}))) {
        return sink.append(source);
      };
  controlled_input.source_text = backend_source_recipe::materialize(
      canonical_recipe, CanonicalControlledSource.size(),
      controlled_guarded_upper);
  const std::size_t controlled_capacity =
      controlled_input.source_text.capacity();
  rund::kernel::LoweringArtifact controlled =
      MetalControlledMapArtifact(std::move(controlled_input), controlled_plan);
  if (!controlled.ok || controlled.source_text.size() != controlled_upper ||
      controlled.source_text.capacity() != controlled_capacity ||
      controlled.source_text.find("artifact_variant=controlled") ==
          std::string::npos ||
      controlled.source_text.find("_controlled(") == std::string::npos) {
    return false;
  }
  controlled.source_text = PipelinePrivateMetalSource(
      std::move(controlled.source_text), controlled_guarded_upper);
  return controlled.source_text.size() == controlled_guarded_upper &&
         controlled.source_text.capacity() == controlled_capacity;
#else
  return true;
#endif
}

} // namespace node_accel_contract
