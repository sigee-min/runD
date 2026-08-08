#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "../../../../src/accel/vulkan/shader/cache.hpp"
#include "../../../../src/accel/backend/token.hpp"
#include "../../../../src/accel/kernel/step/map/stride.hpp"
#include "../../../../src/accel/source/hash.hpp"
#include "../../../../src/accel/vulkan/adapter/api.hpp"
#include "../../../../src/accel/vulkan/adapter/state.hpp"
#include "../../../../src/accel/vulkan/cached/index.hpp"
#include "../../../../src/accel/vulkan/cached/pipeline.hpp"
#include "../../../../src/accel/vulkan/collective/pipeline.hpp"
#include "../../../../src/accel/vulkan/descriptor.hpp"
#include "../../../../src/accel/vulkan/kernel/pipeline/source.hpp"
#include "../../../../src/accel/vulkan/shader/module.hpp"

#include <kernel/program/compute/binding/model.hpp>
#include <node/accel/pick.hpp>
#include <rund/counter.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Handle>
[[nodiscard]] Handle FakeHandle(const std::uintptr_t value) noexcept {
  if constexpr (std::is_pointer_v<Handle>) {
    return reinterpret_cast<Handle>(value);
  } else {
    return static_cast<Handle>(value);
  }
}

[[nodiscard]] int DescriptorRange() {
  using namespace rund::node::accel::detail;
  VulkanAdapter adapter{};
  adapter.storage_align = 16u;
  adapter.storage_limit = 128u;

  VkDescriptorBufferInfo info{
      .buffer = FakeHandle<VkBuffer>(1u), .offset = 16u, .range = 112u};
  if (!ValidStorage(adapter, info)) {
    return 30;
  }
  info.offset = 4u;
  if (ValidStorage(adapter, info) ||
      WriteVulkanStorageDescriptorSet(adapter, FakeHandle<VkDescriptorSet>(2u),
                                      &info, 1u)) {
    return 31;
  }
  info.offset = 0u;
  info.range = 129u;
  if (ValidStorage(adapter, info) ||
      WriteVulkanStorageDescriptorSet(adapter, FakeHandle<VkDescriptorSet>(2u),
                                      &info, 1u)) {
    return 32;
  }
  info.range = VK_WHOLE_SIZE;
  if (ValidStorage(adapter, info)) {
    return 33;
  }

  const VulkanBuffer owner{.buffer = FakeHandle<VkBuffer>(3u), .bytes = 64u};
  const VulkanStorageBinding overflow{&owner, 48u, 32u};
  if (WriteVulkanStorageDescriptorSet(adapter, FakeHandle<VkDescriptorSet>(4u),
                                      &overflow, 1u)) {
    return 34;
  }

  const rund::kernel::ResidentBufferRef strided{
      .bytes = 400u,
      .offset_bytes = 4u,
      .element_bytes = 4u,
      .stride_bytes = 20u,
      .count = 20u,
      .usage = rund::kernel::kResidentUsageRead,
  };
  StorageRange range{};
  if (PlanStorage(adapter, strided, 0u, strided.count, range)) {
    return 35;
  }
  std::uint64_t begin = 0u;
  std::uint64_t pages = 0u;
  constexpr std::array expected_base{0u, 144u, 272u};
  constexpr std::array expected_count{7u, 7u, 6u};
  constexpr std::array expected_bytes{128u, 124u, 116u};
  while (begin < strided.count) {
    if (pages >= expected_count.size() ||
        !PlanStoragePage(adapter, strided, begin, range) ||
        range.base != expected_base[pages] ||
        range.count != expected_count[pages] ||
        range.bytes != expected_bytes[pages] ||
        range.bytes > adapter.storage_limit) {
      return 36;
    }
    begin += range.count;
    ++pages;
  }
  if (begin != strided.count || pages != expected_count.size() ||
      PlanStoragePage(adapter, strided, strided.count, range) ||
      !PlanStorage(adapter, strided, 0u, expected_count[0], range)) {
    return 37;
  }
  return 0;
}

[[nodiscard]] int MapBias() {
  using namespace rund::node::accel::detail;
  rund::kernel::LoweringArtifact artifact{};
  artifact.key.api = rund::kernel::ComputeApi::Vulkan;
  artifact.kind = rund::kernel::LoweringArtifactKind::VulkanSource;
  artifact.metadata.binding_accesses = {
      rund::kernel::ComputeBindingAccess::Read,
      rund::kernel::ComputeBindingAccess::Read,
      rund::kernel::ComputeBindingAccess::Write};
  artifact.metadata.binding_names = {"input1", "input10", "out"};
  artifact.metadata.input_element_bytes = {4u, 4u};
  artifact.metadata.output_element_bytes = {4u};
  artifact.metadata.read_count = 2u;
  artifact.metadata.write_count = 1u;
  artifact.metadata.ok = true;
  artifact.metadata.reason = "ok";
  artifact.source_text = "const uint RundBase_read_696e70757431 = 0u;\n"
                         "const uint RundStride_read_696e70757431 = 4u;\n"
                         "const uint RundBase_read_696e7075743130 = 0u;\n"
                         "const uint RundStride_read_696e7075743130 = 4u;\n"
                         "const uint RundBase_write_6f7574 = 0u;\n"
                         "const uint RundStride_write_6f7574 = 4u;\n"
                         "load(RundBase_read_696e70757431 + "
                         "load(RundBase_read_696e7075743130 + gid * "
                         "RundStride_read_696e7075743130) * "
                         "RundStride_read_696e70757431);\n"
                         "store(RundBase_write_6f7574 + gid * "
                         "RundStride_write_6f7574);\n";
  artifact.ok = true;
  artifact.reason = "ok";
  artifact.source_text_upper_bytes = artifact.source_text.size();

  const rund::kernel::ComputePlan plan{.api = rund::kernel::ComputeApi::Vulkan,
                                       .input_buffer_count = 2u,
                                       .output_buffer_count = 1u};
  const std::array inputs{
      rund::kernel::ResidentBufferRef{.bytes = 64u,
                                      .offset_bytes = 4u,
                                      .element_bytes = 4u,
                                      .stride_bytes = 8u,
                                      .count = 4u,
                                      .usage =
                                          rund::kernel::kResidentUsageRead},
      rund::kernel::ResidentBufferRef{.bytes = 64u,
                                      .offset_bytes = 8u,
                                      .element_bytes = 4u,
                                      .stride_bytes = 4u,
                                      .count = 4u,
                                      .usage =
                                          rund::kernel::kResidentUsageRead},
  };
  const rund::kernel::ResidentBufferRef output{
      .bytes = 64u,
      .offset_bytes = 12u,
      .element_bytes = 4u,
      .stride_bytes = 4u,
      .count = 4u,
      .usage = rund::kernel::kResidentUsageWrite};
  rund::kernel::BindingSet bindings{};
  bindings.resident_inputs =
      rund::kernel::ResidentBindingRange{.refs = inputs.data(),
                                         .storage_count = inputs.size(),
                                         .count = inputs.size()};
  bindings.resident_outputs = rund::kernel::ResidentBindingRange{
      .refs = &output, .storage_count = 1u, .count = 1u};

  const rund::kernel::LoweringArtifact specialized =
      SpecializeMap(artifact, plan, bindings, 16u);
  if (!specialized.ok ||
      specialized.source_text.find(
          "const uint RundStride_read_696e70757431 = 8u;") ==
          std::string::npos ||
      specialized.source_text.find(
          "const uint RundBase_read_696e70757431 = 4u;") == std::string::npos ||
      specialized.source_text.find(
          "const uint RundBase_read_696e7075743130 = 8u;") ==
          std::string::npos ||
      specialized.source_text.find(
          "const uint RundBase_read_696e7075743130 = 4u;") !=
          std::string::npos ||
      specialized.source_text.find("const uint RundBase_write_6f7574 = 12u;") ==
          std::string::npos) {
    std::fprintf(stderr, "map bias failed reason=%s source=%s\n",
                 specialized.reason, specialized.source_text.c_str());
    return 35;
  }
  if (SpecializeMap(artifact, plan, bindings, 0u).ok) {
    return 36;
  }
  return 0;
}

[[nodiscard]] int VulkanCollectiveSourceIdentityContract() {
  rund::AccelPolicy policy{};
  policy.preferred[0] = rund::AccelApi::Vulkan;
  policy.preferred_count = 1u;
  policy.allow_fake = false;
  const rund::AccelDevice pick = rund::node::accel::PickAccel(policy);
  if (!pick.check.ok) {
    return 0;
  }
  using namespace rund::node::accel::detail;
  const std::shared_ptr<PickToken> token = AdmitPick(pick);
  const rund::AccelDevice *const raw = token == nullptr ? nullptr : &token->raw;
  VulkanAdapter *const adapter =
      raw == nullptr ? nullptr : CheckedVulkanAdapter(*raw);
  if (adapter == nullptr) {
    return 15;
  }
  const rund::kernel::ComputePlan plan{
      .op_hash_hi = 0x8fcb4ad7211038c1ull,
      .op_hash_lo = 0x72970c9c2b1676c9ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact first_artifact{
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text =
          "#version 450\nlayout(local_size_x=1) in;\nvoid main() {}\n",
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact second_artifact{
      .kind = rund::kernel::LoweringArtifactKind::VulkanSource,
      .source_text = "#version 450\nlayout(local_size_x=1) in;\n"
                     "void main() { uint x = gl_GlobalInvocationID.x; }\n",
      .ok = true,
      .reason = "ok",
  };
  std::lock_guard lock{adapter->mutex};
  const std::uint64_t compile_begin = adapter->pipeline_compile_count;
  const std::uint64_t hit_begin = adapter->pipeline_cache_hit_count;
  const std::uint64_t module_begin = adapter->shader_module_create_count;
  VulkanCollectivePipeline *const first =
      AcquireVulkanCollectivePipeline(*adapter, 1u, 0u, plan, first_artifact);
  VulkanCollectivePipeline *const repeated =
      AcquireVulkanCollectivePipeline(*adapter, 1u, 0u, plan, first_artifact);
  VulkanCollectivePipeline *const distinct =
      AcquireVulkanCollectivePipeline(*adapter, 1u, 0u, plan, second_artifact);
  VulkanCollectivePipeline *const pushed = AcquireVulkanCollectivePipeline(
      *adapter, 1u, sizeof(std::uint32_t), plan, first_artifact);
  VulkanCollectivePipeline *const wider =
      AcquireVulkanCollectivePipeline(*adapter, 2u, 0u, plan, first_artifact);
  const rund::kernel::ComputePlan map_plan{
      .op_hash_hi = 0xdb530c5f26b24c11ull,
      .op_hash_lo = 0x8e4410b17a77db21ull,
      .api = rund::kernel::ComputeApi::Vulkan,
      .output_buffer_count = 1u,
      .ok = true,
      .reason = "ok",
  };
  const rund::kernel::LoweringArtifact map_artifact{
      .key =
          {
              .api = rund::kernel::ComputeApi::Vulkan,
              .op_hash_hi = map_plan.op_hash_hi,
              .op_hash_lo = map_plan.op_hash_lo,
          },
      .source_text =
          "#version 450\nlayout(local_size_x=1) in;\nvoid main() {}\n",
      .ok = true,
      .reason = "ok",
  };
  VulkanCachedPipeline *const map =
      AcquireVulkanCachedPipeline(*adapter, map_plan, map_artifact);
  VulkanCachedPipeline *const map_repeated =
      AcquireVulkanCachedPipeline(*adapter, map_plan, map_artifact);
  VulkanSpecialization invalid_specialization{};
  invalid_specialization.count =
      static_cast<std::uint32_t>(invalid_specialization.values.size() + 1u);
  VulkanCollectivePipeline *const invalid = AcquireVulkanCollectivePipeline(
      *adapter, 1u, 0u, plan, first_artifact, invalid_specialization);
  if (first == nullptr) {
    return 16;
  }
  if (repeated != first) {
    return 17;
  }
  if (distinct == nullptr) {
    return 18;
  }
  if (distinct == first) {
    return 19;
  }
  if (pushed == nullptr || pushed == first ||
      pushed->push_bytes != sizeof(std::uint32_t)) {
    return 22;
  }
  if (wider == nullptr || wider == first || wider == pushed ||
      wider->descriptor_count != 2u) {
    return 25;
  }
  if (map == nullptr || map_repeated != map) {
    return 38;
  }
  if (invalid != nullptr || adapter->shader_module_current != 0u) {
    return 39;
  }
  if (::rund::detail::counter::Delta(compile_begin,
                                     adapter->pipeline_compile_count) != 5u) {
    return 20;
  }
  if (::rund::detail::counter::Delta(hit_begin,
                                     adapter->pipeline_cache_hit_count) != 2u) {
    return 21;
  }
  if (adapter->shader_module_current != 0u ||
      adapter->shader_module_peak != 1u ||
      ::rund::detail::counter::Delta(
          module_begin, adapter->shader_module_create_count) != 6u) {
    return 37;
  }
  return 0;
}

} // namespace
#endif

int RunComputeVulkanCacheContract() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using rund::node::accel::detail::VulkanAdapter;
  using rund::node::accel::detail::VulkanCachedPipeline;
  using rund::node::accel::detail::VulkanCollectivePipeline;
  using rund::node::accel::detail::VulkanModule;
  using rund::node::accel::detail::VulkanShader;

  static_assert(!std::is_copy_constructible_v<VulkanModule>);
  static_assert(!std::is_copy_assignable_v<VulkanModule>);
  static_assert(std::is_nothrow_move_constructible_v<VulkanModule>);
  static_assert(std::is_nothrow_move_assignable_v<VulkanModule>);
  static_assert(!std::is_copy_constructible_v<VulkanCachedPipeline>);
  static_assert(!std::is_copy_assignable_v<VulkanCachedPipeline>);
  static_assert(std::is_nothrow_move_constructible_v<VulkanCachedPipeline>);
  static_assert(std::is_nothrow_move_assignable_v<VulkanCachedPipeline>);
  static_assert(!std::is_copy_constructible_v<VulkanCollectivePipeline>);
  static_assert(!std::is_copy_assignable_v<VulkanCollectivePipeline>);
  static_assert(std::is_nothrow_move_constructible_v<VulkanCollectivePipeline>);
  static_assert(std::is_nothrow_move_assignable_v<VulkanCollectivePipeline>);

  constexpr std::uint64_t telemetry_hash_hi = 0x706970652e74656cull;
  constexpr std::uint64_t telemetry_hash_lo = 0x656d657472792e31ull;
  constexpr std::uint64_t profile_hash_hi = 0x706970652e70726full;
  constexpr std::uint64_t profile_hash_lo = 0x66696c652e763100ull;
  const std::string_view telemetry_source =
      rund::node::accel::detail::VulkanTelemetrySourceText();
  const std::string profile_source =
      rund::node::accel::detail::VulkanProfileSource();
  const auto telemetry_plan = rund::node::accel::detail::VulkanTelemetryPlan();
  const auto profile_plan = rund::node::accel::detail::VulkanProfilePlan();
  if (telemetry_source.empty() || profile_source.empty() ||
      telemetry_source == profile_source ||
      rund::node::accel::detail::VulkanProfileSourceBytes() !=
          profile_source.size() ||
      telemetry_plan.op_hash_hi != telemetry_hash_hi ||
      telemetry_plan.op_hash_lo != telemetry_hash_lo ||
      profile_plan.op_hash_hi != profile_hash_hi ||
      profile_plan.op_hash_lo != profile_hash_lo ||
      telemetry_plan.api != rund::kernel::ComputeApi::Vulkan ||
      profile_plan.api != rund::kernel::ComputeApi::Vulkan ||
      !telemetry_plan.ok || !profile_plan.ok) {
    return 40;
  }

  if (const int descriptor = DescriptorRange(); descriptor != 0) {
    return descriptor;
  }
  if (const int bias = MapBias(); bias != 0) {
    return bias;
  }

  VulkanAdapter adapter{};
  adapter.pipelines.push_back(VulkanCachedPipeline{});
  VulkanCachedPipeline *const first = &adapter.pipelines.front();
  for (std::size_t index = 0u; index < 128u; ++index) {
    adapter.pipelines.push_back(VulkanCachedPipeline{});
  }
  if (&adapter.pipelines.front() != first) {
    return 1;
  }

  adapter.collective_pipelines.emplace_back();
  adapter.collective_pipelines.emplace_back();
  VulkanCollectivePipeline &used = adapter.collective_pipelines.front();
  VulkanCollectivePipeline &idle = adapter.collective_pipelines.back();
  used.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  idle.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  used.next_descriptor_slot = 7u;
  idle.next_descriptor_slot = 11u;
  used.descriptor_sets.resize(3u, VK_NULL_HANDLE);
  BeginVulkanCollectiveDescriptorEpoch(adapter);
  if (used.next_descriptor_slot != 7u || idle.next_descriptor_slot != 11u) {
    return 26;
  }
  PrepareVulkanCollectiveDescriptorSlots(adapter, used);
  if (used.next_descriptor_slot != 0u || used.reusable_descriptor_count != 3u ||
      idle.next_descriptor_slot != 11u) {
    return 27;
  }
  used.next_descriptor_slot = 2u;
  PrepareVulkanCollectiveDescriptorSlots(adapter, used);
  if (used.next_descriptor_slot != 2u) {
    return 28;
  }
  adapter.pipeline_index->descriptor_epoch =
      std::numeric_limits<std::uint64_t>::max();
  used.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  idle.descriptor_epoch = adapter.pipeline_index->descriptor_epoch;
  BeginVulkanCollectiveDescriptorEpoch(adapter);
  if (adapter.pipeline_index->descriptor_epoch != 1u ||
      used.descriptor_epoch != 0u || idle.descriptor_epoch != 0u) {
    return 29;
  }

  const rund::kernel::ArtifactKey exact_key{
      .api = rund::kernel::ComputeApi::Vulkan,
      .op_hash_hi = 17u,
      .op_hash_lo = 29u,
      .canonical_ir_hash_hi = 41u,
      .canonical_ir_hash_lo = 53u,
  };
  const rund::kernel::LoweringArtifact exact_artifact{
      .key = exact_key,
      .source_text = "exact-source-a",
  };
  const rund::kernel::LoweringArtifact colliding_artifact{
      .key = exact_key,
      .source_text = "exact-source-b",
  };
  rund::kernel::ArtifactKey distinct_key = exact_key;
  distinct_key.op_hash_lo += 1u;
  const rund::kernel::LoweringArtifact distinct_artifact{
      .key = distinct_key,
      .source_text = exact_artifact.source_text,
  };
  const auto handle = [](const std::uintptr_t value) {
    return reinterpret_cast<void *>(value);
  };
  VulkanCachedPipeline exact_pipeline{};
  exact_pipeline.key = exact_key;
  exact_pipeline.source_hash =
      rund::node::accel::detail::SourceHash(exact_artifact.source_text);
  exact_pipeline.source = exact_artifact.source_text;
  exact_pipeline.input_buffer_count = 2u;
  exact_pipeline.output_buffer_count = 1u;
  exact_pipeline.descriptor_set_layout =
      reinterpret_cast<VkDescriptorSetLayout>(handle(1u));
  exact_pipeline.pipeline_layout =
      reinterpret_cast<VkPipelineLayout>(handle(2u));
  exact_pipeline.pipeline = reinterpret_cast<VkPipeline>(handle(3u));
  const std::uint64_t exact_hash =
      rund::node::accel::detail::SourceHash(exact_artifact.source_text);
  if (!VulkanCachedPipelineMatches(exact_pipeline, exact_artifact, exact_hash,
                                   2u, 1u) ||
      VulkanCachedPipelineMatches(exact_pipeline, colliding_artifact,
                                  exact_hash, 2u, 1u) ||
      VulkanCachedPipelineMatches(exact_pipeline, distinct_artifact, exact_hash,
                                  2u, 1u) ||
      VulkanCachedPipelineMatches(exact_pipeline, exact_artifact, exact_hash,
                                  1u, 1u) ||
      VulkanCachedPipelineMatches(exact_pipeline, exact_artifact, exact_hash,
                                  2u, 2u)) {
    return 23;
  }
  VulkanCachedPipeline moved_pipeline{std::move(exact_pipeline)};
  if (exact_pipeline.pipeline != VK_NULL_HANDLE ||
      exact_pipeline.pipeline_layout != VK_NULL_HANDLE ||
      exact_pipeline.descriptor_set_layout != VK_NULL_HANDLE ||
      moved_pipeline.pipeline == VK_NULL_HANDLE ||
      moved_pipeline.pipeline_layout == VK_NULL_HANDLE ||
      moved_pipeline.descriptor_set_layout == VK_NULL_HANDLE) {
    return 24;
  }

  using namespace rund::node::accel::detail;
  ClearValidatedVulkanSpirvCache();
  auto words = std::make_shared<const std::vector<std::uint32_t>>(
      std::vector<std::uint32_t>{0x07230203u, 1u, 2u, 3u});
  const VulkanShader original{.words = words, .hash = 17u};
  CacheValidatedVulkanSpirv("compiler-a", "validator-a", "source-a", original);
  if (ValidatedVulkanSpirvCacheSize() != 1u ||
      ValidatedVulkanSpirvCacheBytes() == 0u ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity) {
    return 2;
  }

  VulkanShader found{};
  if (!FindValidatedVulkanSpirv("compiler-a", "validator-a", "source-a",
                                found) ||
      found.words != words || found.hash != original.hash) {
    return 3;
  }
  if (FindValidatedVulkanSpirv("compiler-b", "validator-a", "source-a",
                               found) ||
      FindValidatedVulkanSpirv("compiler-a", "validator-b", "source-a",
                               found) ||
      FindValidatedVulkanSpirv("compiler-a", "validator-a", "source-b",
                               found)) {
    return 4;
  }

  for (std::size_t index = 0u; index < kVulkanSpirvCacheCapacity; ++index) {
    const std::string source = "bounded-source-" + std::to_string(index);
    CacheValidatedVulkanSpirv("compiler-a", "validator-a", source, original);
  }
  if (ValidatedVulkanSpirvCacheSize() != kVulkanSpirvCacheCapacity ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity ||
      FindValidatedVulkanSpirv("compiler-a", "validator-a", "source-a",
                               found) ||
      !FindValidatedVulkanSpirv(
          "compiler-a", "validator-a",
          "bounded-source-" + std::to_string(kVulkanSpirvCacheCapacity - 1u),
          found) ||
      found.words != words) {
    return 5;
  }
  ClearValidatedVulkanSpirvCache();
  if (ValidatedVulkanSpirvCacheSize() != 0u ||
      ValidatedVulkanSpirvCacheBytes() != 0u || found.words != words ||
      found.words->size() != 4u || found.words->front() != 0x07230203u) {
    return 6;
  }
  const std::string oversized_source(kVulkanSpirvCacheByteCapacity, 'x');
  CacheValidatedVulkanSpirv("compiler-a", "validator-a", oversized_source,
                            original);
  if (ValidatedVulkanSpirvCacheSize() != 0u ||
      ValidatedVulkanSpirvCacheBytes() != 0u) {
    return 7;
  }

  constexpr std::size_t kByteEntryCount = 20u;
  constexpr std::size_t kByteSourceSize = 1u * 1024u * 1024u;
  constexpr std::string_view kByteCompiler = "byte-compiler";
  constexpr std::string_view kByteValidator = "byte-validator";
  const std::size_t byte_entry_size = kByteSourceSize + kByteCompiler.size() +
                                      kByteValidator.size() +
                                      words->size() * sizeof(std::uint32_t);
  const std::size_t expected_byte_entries =
      std::min(kVulkanSpirvCacheCapacity,
               kVulkanSpirvCacheByteCapacity / byte_entry_size);
  std::string byte_source(kByteSourceSize, 'x');
  for (std::size_t index = 0u; index < kByteEntryCount; ++index) {
    byte_source.back() = static_cast<char>('a' + index);
    CacheValidatedVulkanSpirv(kByteCompiler, kByteValidator, byte_source,
                              original);
  }
  std::string first_byte_source(kByteSourceSize, 'x');
  first_byte_source.back() = 'a';
  std::string last_byte_source(kByteSourceSize, 'x');
  last_byte_source.back() = static_cast<char>('a' + kByteEntryCount - 1u);
  if (ValidatedVulkanSpirvCacheSize() != expected_byte_entries ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity ||
      FindValidatedVulkanSpirv(kByteCompiler, kByteValidator, first_byte_source,
                               found) ||
      !FindValidatedVulkanSpirv(kByteCompiler, kByteValidator, last_byte_source,
                                found) ||
      found.words != words) {
    return 8;
  }
  ClearValidatedVulkanSpirvCache();

  constexpr std::size_t kThreadCount = 8u;
  constexpr std::size_t kExactEntriesPerThread = 16u;
  std::atomic_bool concurrent_exact{true};
  std::array<std::thread, kThreadCount> exact_threads{};
  for (std::size_t thread_index = 0u; thread_index < kThreadCount;
       ++thread_index) {
    exact_threads[thread_index] =
        std::thread{[thread_index, &original, &words, &concurrent_exact]() {
          const std::string compiler =
              "concurrent-compiler-" + std::to_string(thread_index);
          const std::string validator =
              "concurrent-validator-" + std::to_string(thread_index);
          for (std::size_t entry_index = 0u;
               entry_index < kExactEntriesPerThread; ++entry_index) {
            const std::string source = "concurrent-source-" +
                                       std::to_string(thread_index) + "-" +
                                       std::to_string(entry_index);
            CacheValidatedVulkanSpirv(compiler, validator, source, original);
            VulkanShader concurrent_found{};
            if (!FindValidatedVulkanSpirv(compiler, validator, source,
                                          concurrent_found) ||
                concurrent_found.words != words ||
                concurrent_found.hash != original.hash ||
                FindValidatedVulkanSpirv(compiler + "-other", validator, source,
                                         concurrent_found) ||
                FindValidatedVulkanSpirv(compiler, validator + "-other", source,
                                         concurrent_found) ||
                FindValidatedVulkanSpirv(compiler, validator, source + "-other",
                                         concurrent_found)) {
              concurrent_exact.store(false, std::memory_order_relaxed);
            }
          }
        }};
  }
  for (std::thread &thread : exact_threads) {
    thread.join();
  }
  if (!concurrent_exact.load(std::memory_order_relaxed) ||
      ValidatedVulkanSpirvCacheSize() !=
          kThreadCount * kExactEntriesPerThread ||
      ValidatedVulkanSpirvCacheBytes() == 0u ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity) {
    return 9;
  }

  ClearValidatedVulkanSpirvCache();
  constexpr std::size_t kBoundedEntriesPerThread = 64u;
  std::array<std::thread, kThreadCount> bounded_threads{};
  for (std::size_t thread_index = 0u; thread_index < kThreadCount;
       ++thread_index) {
    bounded_threads[thread_index] = std::thread{[thread_index, &original]() {
      for (std::size_t entry_index = 0u; entry_index < kBoundedEntriesPerThread;
           ++entry_index) {
        const std::string source = "concurrent-bounded-source-" +
                                   std::to_string(thread_index) + "-" +
                                   std::to_string(entry_index);
        CacheValidatedVulkanSpirv("concurrent-compiler", "concurrent-validator",
                                  source, original);
      }
    }};
  }
  for (std::thread &thread : bounded_threads) {
    thread.join();
  }
  if (ValidatedVulkanSpirvCacheSize() != kVulkanSpirvCacheCapacity ||
      ValidatedVulkanSpirvCacheBytes() == 0u ||
      ValidatedVulkanSpirvCacheBytes() > kVulkanSpirvCacheByteCapacity) {
    return 10;
  }
  ClearValidatedVulkanSpirvCache();

  constexpr std::string_view kToolSource = R"(
#version 450
layout(local_size_x = 1) in;
void main() {}
)";
  rund::kernel::ComputePlan tool_plan{};
  rund::kernel::LoweringArtifact tool_artifact{};
  tool_artifact.source_text = kToolSource;
  VulkanAdapter tool_adapter{};
  tool_adapter.glslang_validator_path =
      "/rund-node-intentionally-missing-compiler";
  VulkanShader tool_shader{};
  if (CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                   tool_shader) ||
      ValidatedVulkanSpirvCacheSize() != 0u) {
    return 11;
  }
#if defined(RUND_NODE_TEST_GLSLANG_VALIDATOR_PATH)
  tool_adapter.glslang_validator_path = RUND_NODE_TEST_GLSLANG_VALIDATOR_PATH;
  tool_adapter.spirv_val_path.clear();
  if (!CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                    tool_shader) ||
      ValidatedVulkanSpirvCacheSize() != 1u) {
    return 12;
  }
  VulkanShader published = tool_shader;
  tool_shader = {};
  if (!CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                    tool_shader) ||
      tool_shader.words != published.words ||
      ValidatedVulkanSpirvCacheSize() != 1u) {
    return 13;
  }
  ClearValidatedVulkanSpirvCache();
  tool_adapter.spirv_val_path = "/rund-node-intentionally-missing-validator";
  tool_shader = {};
  if (CompileVulkanSourceWithTools(tool_adapter, tool_plan, tool_artifact,
                                   tool_shader) ||
      ValidatedVulkanSpirvCacheSize() != 0u ||
      ValidatedVulkanSpirvCacheBytes() != 0u) {
    return 14;
  }
#endif
  if (const int source_identity = VulkanCollectiveSourceIdentityContract();
      source_identity != 0) {
    return source_identity;
  }
#endif
  return 0;
}
