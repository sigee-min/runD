#include <accel/context/buffer/descriptor.hpp>

#include "local.hpp"
#include <node/accel/buffer.hpp>
#include <node/accel/context.hpp>

#include "src/accel/context/admission.hpp"

#if defined(RUND_NODE_HAVE_VULKAN_SDK)
#include "src/accel/vulkan/buffer/resident/batch/local.hpp"
#include "src/accel/vulkan/buffer/transfer/range.hpp"
#endif

#include <algorithm>
#include <vector>

namespace {

[[nodiscard]] bool VulkanUploadPreservationIsMinimal() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using rund::node::accel::detail::ResolveVulkanTransferRange;
  using rund::node::accel::detail::ResolveVulkanUploadPreservation;
  using rund::node::accel::detail::VulkanTransferRange;
  using rund::node::accel::detail::VulkanUploadPreservation;
  const auto resolve = [](const std::uint64_t offset, const std::uint64_t bytes,
                          const std::uint64_t semantic_bytes,
                          VulkanUploadPreservation &preservation) {
    VulkanTransferRange range{};
    return ResolveVulkanTransferRange(offset, bytes, 32u, range) &&
           ResolveVulkanUploadPreservation(range, offset, bytes, semantic_bytes,
                                           preservation);
  };

  VulkanUploadPreservation aligned{};
  VulkanUploadPreservation left{};
  VulkanUploadPreservation right{};
  VulkanUploadPreservation both{};
  VulkanUploadPreservation same_word{};
  VulkanUploadPreservation padded{};
  return resolve(4u, 8u, 32u, aligned) && aligned.region_count == 0u &&
         aligned.padding_bytes == 0u && resolve(5u, 7u, 32u, left) &&
         left.region_count == 1u && left.regions[0].srcOffset == 4u &&
         left.regions[0].dstOffset == 0u && left.regions[0].size == 4u &&
         resolve(4u, 7u, 32u, right) && right.region_count == 1u &&
         right.regions[0].srcOffset == 8u && right.regions[0].dstOffset == 4u &&
         right.regions[0].size == 4u && resolve(5u, 6u, 32u, both) &&
         both.region_count == 2u && both.regions[0].srcOffset == 4u &&
         both.regions[1].srcOffset == 8u && resolve(1u, 1u, 32u, same_word) &&
         same_word.region_count == 1u && same_word.regions[0].srcOffset == 0u &&
         resolve(0u, 7u, 7u, padded) && padded.region_count == 0u &&
         padded.padding_offset == 7u && padded.padding_bytes == 1u;
#else
  return true;
#endif
}

[[nodiscard]] bool VulkanBatchStagingIsBounded() {
#if defined(RUND_NODE_HAVE_VULKAN_SDK)
  using namespace rund::node::accel::detail;
  constexpr VkDeviceSize budget = 8u;
  std::uint64_t consumed = 0u;
  std::vector<TransferSlice> slices;
  while (consumed < 20u) {
    TransferSlice slice{};
    if (!next_slice(1u, 20u, consumed, budget, slice)) {
      return false;
    }
    slices.push_back(slice);
    consumed += slice.bytes;
  }
  if (transfer_budget(0u) != kVulkanTransferAlignment || slices.size() != 3u ||
      slices[0u].offset != 1u || slices[0u].bytes != 7u ||
      slices[1u].offset != 8u || slices[1u].bytes != 8u ||
      slices[2u].offset != 16u || slices[2u].bytes != 5u) {
    return false;
  }
  std::vector<DownloadPlan> plans(slices.size());
  for (std::size_t index = 0u; index < slices.size(); ++index) {
    VulkanTransferRange range{};
    if (!ResolveVulkanTransferRange(slices[index].offset, slices[index].bytes,
                                    24u, range) ||
        range.bytes > budget) {
      return false;
    }
    plans[index].range = range;
  }
  const std::vector<BatchChunk> chunks = batch_chunks(plans, budget);
  return chunks.size() == plans.size() &&
         std::all_of(chunks.begin(), chunks.end(), [](const BatchChunk chunk) {
           return chunk.end == chunk.begin + 1u && chunk.bytes == budget;
         });
#else
  return true;
#endif
}

[[nodiscard]] bool TransferCauseIsPreserved() {
  using rund::node::accel::detail::TransferCheckFrom;
  const rund::AccelCheck lost = TransferCheckFrom(
      {false, "compute_device_lost"}, "accel_buffer_download_overflow");
  const rund::AccelCheck internal =
      TransferCheckFrom({false, "accel_vulkan_transfer_invalid"},
                        "accel_buffer_download_overflow");
  const rund::AccelCheck overflow =
      TransferCheckFrom({false, "accel_buffer_download_overflow"},
                        "accel_buffer_download_overflow");
  return !lost.ok && std::string_view{lost.reason} == "compute_device_lost" &&
         !internal.ok &&
         std::string_view{internal.reason} == "accel_vulkan_transfer_invalid" &&
         !overflow.ok &&
         std::string_view{overflow.reason} == "accel_buffer_download_overflow";
}

} // namespace

namespace node_accel_contract {

bool PublicContextApiCreatesAndTransfers() {
  namespace ctx = node_accel_contract::context;
  if (!VulkanUploadPreservationIsMinimal() || !VulkanBatchStagingIsBounded() ||
      !TransferCauseIsPreserved()) {
    return false;
  }
  const ctx::State state = ctx::OpenState();
  if (!state.available) {
    return state.unavailable_ok;
  }

  rund::AccelBufferDesc zero_count_desc = state.typed_desc;
  zero_count_desc.count = 0u;
  if (!ctx::CheckReason(
          rund::node::accel::CreateAccelBuffer(state.context, zero_count_desc)
              .check,
          "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBufferDesc zero_width_desc = state.typed_desc;
  zero_width_desc.scalar_width_bytes = 0u;
  if (!ctx::CheckReason(
          rund::node::accel::CreateAccelBuffer(state.context, zero_width_desc)
              .check,
          "accel_context_buffer_invalid")) {
    return false;
  }

  rund::AccelBufferDesc create_overflow_desc = state.typed_desc;
  create_overflow_desc.scalar_width_bytes =
      std::numeric_limits<std::uint64_t>::max();
  create_overflow_desc.count = 2u;
  if (!ctx::CheckReason(rund::node::accel::CreateAccelBuffer(
                            state.context, create_overflow_desc)
                            .check,
                        "accel_context_buffer_overflow")) {
    return false;
  }

  if (!state.created.check.ok ||
      std::string_view{state.created.reason} != "ok" ||
      state.created.byte_extent != state.typed.byte_extent ||
      state.created.context_id != state.context.id ||
      !rund::node::test::SameOwner(state.created.owner, state.context.owner) ||
      rund::node::test::SameOwner(state.created.owner, state.pick.owner) ||
      state.created.buffer.bytes != state.typed.byte_extent ||
      state.created.buffer.handle != state.created.handle ||
      !rund::node::test::SameOwner(state.created.buffer.owner,
                                   state.context.owner) ||
      state.created.resident.id != state.created.buffer.id ||
      state.created.resident.bytes != state.created.buffer.bytes ||
      state.created.resident.element_bytes !=
          state.typed_desc.scalar_width_bytes ||
      state.created.resident.count != state.typed_desc.count) {
    return false;
  }

  std::array<std::uint32_t, 8u> upload_data{1u, 1u, 2u, 3u, 5u, 8u, 13u, 21u};
  std::array<std::uint32_t, 8u> download_data{};
  rund::node::accel::ResetRuntimeStats(state.pick);
  return rund::node::accel::UploadAccelBuffer(state.context, state.created,
                                              upload_data.data(),
                                              sizeof(upload_data))
             .ok &&
         rund::node::accel::DownloadAccelBuffer(state.context, state.created,
                                                download_data.data(),
                                                sizeof(download_data))
             .ok &&
         upload_data == download_data &&
         ctx::RuntimeBytesInclude(
             rund::node::accel::ReadRuntimeStats(state.pick),
             sizeof(upload_data), sizeof(download_data));
}

} // namespace node_accel_contract
