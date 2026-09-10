#include "../internal.hpp"

#include "../../backing.hpp"

namespace rund::compute::detail::sliding_product_detail {

PersistIo perform_persist_io(
    SlidingProductRun &state,
    const residency::execution::SlidingPersist &persist) noexcept {
  PersistIo io{};
  io.started_ns = pipeline_clock();
  io.frame = virtual_host_output_frame(*state.run, persist.frame());
  const std::uint64_t page_offset =
      persist.key().page * state.run->output_payload_bytes;
  const residency::DirtyExtent extent = persist.extent();
  if (io.frame == nullptr || extent.offset < page_offset ||
      extent.bytes != persist.expected_bytes() ||
      extent.bytes > std::numeric_limits<std::size_t>::max()) {
    io.status = Status::fail(Reason::PipelineInvalid);
    return io;
  }
  io.hash_offset = static_cast<std::size_t>(state.run->output_prefix_bytes +
                                            extent.offset - page_offset);
  const VirtualWrite range{
      .offset = extent.offset,
      .bytes = std::span<const std::byte>{
          io.frame + io.hash_offset, static_cast<std::size_t>(extent.bytes)}};
  VirtualBackingTransaction *const transaction =
      VirtualBackingAccess::transaction_provider(*state.output);
  VirtualBackingTransactionToken *const token =
      VirtualBackingAccess::transaction_token(*state.output);
  io.status = transaction != nullptr && token != nullptr
                  ? transaction->stage(
                        *token, std::span<const VirtualWrite>{&range, 1u})
                  : state.output->write(range.offset, range.bytes);
  if (io.status) {
    io.bytes = extent.bytes;
  }
  return io;
}

} // namespace rund::compute::detail::sliding_product_detail
