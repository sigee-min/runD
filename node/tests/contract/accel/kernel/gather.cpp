#include <accel/api.hpp>
#include <accel/device.hpp>

#include "gather/local.hpp"
#include "gather/match/run.hpp"
#include "gather/reject/run.hpp"
#include "src/accel/metal/pipeline/cache.hpp"
#include "src/accel/metal/stats.hpp"
#include <node/accel/pick.hpp>

#include <iostream>
#include <memory>
#include <string>

namespace node_accel_contract {
namespace {

[[nodiscard]] bool MetalSourceLibraryCacheIsBoundedAndLru() {
  using namespace rund::node::accel::detail;
  MetalAdapter adapter{};
  for (std::size_t index = 0u; index < kMetalSourceLibraryCapacity; ++index) {
    std::shared_ptr<void> published =
        PublishMetalSourceLibrary(adapter, "source-" + std::to_string(index),
                                  std::make_shared<std::size_t>(index), 1u);
    if (published == nullptr) {
      return false;
    }
  }
  if (adapter.source_libraries.size() != kMetalSourceLibraryCapacity ||
      LookupMetalSourceLibrary(adapter, "source-0") == nullptr) {
    return false;
  }
  std::shared_ptr<void> published = PublishMetalSourceLibrary(
      adapter, "source-extra", std::make_shared<std::size_t>(17u), 1u);
  return published != nullptr &&
         adapter.source_libraries.size() == kMetalSourceLibraryCapacity &&
         LookupMetalSourceLibrary(adapter, "source-1") == nullptr &&
         LookupMetalSourceLibrary(adapter, "source-0") != nullptr &&
         LookupMetalSourceLibrary(adapter, "source-extra") != nullptr;
}

} // namespace

bool BackendRunsGather(const rund::AccelDevice &pick) {
  return gather::MatchesU32(pick) && gather::MatchesU64(pick) &&
         gather::RejectsOutOfRangeIndex(pick) &&
         gather::RejectsBoundedCountOverflowWithoutMutation(pick);
}

bool RequiredMetalRunsGather() {
  if (!MetalSourceLibraryCacheIsBoundedAndLru()) {
    return false;
  }
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Metal));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Metal);
  }
  if (pick.api != rund::AccelApi::Metal || !gather::MatchesU32(pick)) {
    return false;
  }
  const rund::node::accel::detail::MetalRuntimeStats first =
      rund::node::accel::detail::ReadMetalRuntimeStats(pick);
  if (!first.ok || first.library_compile_count != 1u ||
      first.library_cache_hit_count != 0u || first.shader_compile_ns == 0u) {
    std::cerr << "metal gather u32 library cache mismatch: compile="
              << first.library_compile_count
              << " hit=" << first.library_cache_hit_count
              << " compile_ns=" << first.shader_compile_ns << '\n';
    return false;
  }
  if (!gather::MatchesU64(pick)) {
    return false;
  }
  const rund::node::accel::detail::MetalRuntimeStats second =
      rund::node::accel::detail::ReadMetalRuntimeStats(pick);
  if (!second.ok || second.library_compile_count != 0u ||
      second.library_cache_hit_count != 1u || second.shader_compile_ns != 0u) {
    std::cerr << "metal gather u64 library cache mismatch: compile="
              << second.library_compile_count
              << " hit=" << second.library_cache_hit_count
              << " compile_ns=" << second.shader_compile_ns << '\n';
    return false;
  }
  return gather::RejectsOutOfRangeIndex(pick) &&
         gather::RejectsBoundedCountOverflowWithoutMutation(pick);
}

bool RequiredVulkanRunsGather() {
  const rund::AccelDevice pick =
      rund::node::accel::PickAccel(primitive::Policy(rund::AccelApi::Vulkan));
  if (!pick.check.ok) {
    return primitive::PickUnavailableReasonIsPrecise(pick,
                                                     rund::AccelApi::Vulkan);
  }
  return pick.api == rund::AccelApi::Vulkan && BackendRunsGather(pick) &&
         gather::PreparedRetainsStorage(pick);
}

} // namespace node_accel_contract
