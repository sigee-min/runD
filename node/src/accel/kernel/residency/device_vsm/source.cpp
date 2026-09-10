#include "source.hpp"

#include "source/internal.hpp"

#include <kernel/program/compute/lowering/format.hpp>

#include <algorithm>
#include <limits>
#include <new>
#include <string>

namespace rund::node::accel::detail {
namespace {

using rund::kernel::ComputeApi;
using rund::kernel::LoweringArtifact;
using rund::kernel::LoweringArtifactKind;
using rund::kernel::LoweringArtifactVariant;

} // namespace

bool TransformDeviceVsmSource(LoweringArtifact &artifact,
                              const std::uint64_t input_count,
                              const std::uint64_t output_count) {
  const bool metal = artifact.key.api == ComputeApi::Metal &&
                     artifact.kind == LoweringArtifactKind::MetalSource;
  const bool vulkan = artifact.key.api == ComputeApi::Vulkan &&
                      artifact.kind == LoweringArtifactKind::VulkanSource;
  if (!artifact.ok ||
      artifact.key.variant != LoweringArtifactVariant::Canonical ||
      (!metal && !vulkan) || input_count == 0u || output_count == 0u ||
      input_count != artifact.metadata.read_count ||
      output_count != artifact.metadata.write_count ||
      input_count > std::numeric_limits<std::uint32_t>::max() ||
      output_count > std::numeric_limits<std::uint32_t>::max() ||
      output_count >
          std::numeric_limits<std::uint64_t>::max() - input_count - 1u) {
    return false;
  }
  try {
    LoweringArtifact candidate = artifact;
    if (!device_vsm_source::replace_one(candidate.source_text,
                                        device_vsm_source::CanonicalVariant,
                                        device_vsm_source::DeviceVariant)) {
      return false;
    }
    const std::uint64_t binding = input_count + output_count + 1u;
    if (!(metal ? device_vsm_source::transform_metal(candidate.source_text,
                                                     binding, input_count)
                : device_vsm_source::transform_vulkan(candidate.source_text,
                                                      binding, input_count))) {
      return false;
    }
    candidate.key.variant = LoweringArtifactVariant::DeviceVsm;
    candidate.source_text_upper_bytes = std::max<std::uint64_t>(
        candidate.source_text_upper_bytes, candidate.source_text.size());
    artifact = std::move(candidate);
    return true;
  } catch (const std::bad_alloc &) {
    return false;
  } catch (const std::length_error &) {
    return false;
  }
}

} // namespace rund::node::accel::detail
