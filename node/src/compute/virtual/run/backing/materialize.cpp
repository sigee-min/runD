#include "../backing.hpp"

#include "../../../type.hpp"
#include "../../backing.hpp"

#include <kernel/program/compute/reduce/model.hpp>
#include <kernel/program/compute/window/model.hpp>
#include <rund/compute/pipeline/shape.hpp>
#include <rund/counter.hpp>

#include <array>
#include <cstring>
#include <limits>

namespace rund::compute::detail {
namespace {

template <class T>
void encode_window_identity(const std::uint32_t operation,
                            std::byte *const output) noexcept {
  T value{};
  if (operation == static_cast<std::uint32_t>(kernel::WindowOp::Min)) {
    value = std::numeric_limits<T>::max();
  } else if (operation == static_cast<std::uint32_t>(kernel::WindowOp::Max)) {
    value = std::numeric_limits<T>::lowest();
  }
  std::memcpy(output, &value, sizeof(value));
}

[[nodiscard]] bool window_identity(const Type type,
                                   const std::uint32_t operation,
                                   std::byte *const output) noexcept {
  if (operation != static_cast<std::uint32_t>(kernel::WindowOp::Sum) &&
      operation != static_cast<std::uint32_t>(kernel::WindowOp::Min) &&
      operation != static_cast<std::uint32_t>(kernel::WindowOp::Max)) {
    return false;
  }
  switch (type) {
  case Type::I32:
  case Type::FixedLane32:
    encode_window_identity<std::int32_t>(operation, output);
    return true;
  case Type::U32:
    encode_window_identity<std::uint32_t>(operation, output);
    return true;
  case Type::I64:
  case Type::FixedLane64:
    encode_window_identity<std::int64_t>(operation, output);
    return true;
  case Type::U64:
    encode_window_identity<std::uint64_t>(operation, output);
    return true;
  }
  return false;
}

template <class T>
void encode_reduce_identity(const std::uint32_t operation,
                            std::byte *const output) noexcept {
  T value{};
  if (operation == static_cast<std::uint32_t>(kernel::ReduceOp::Min)) {
    value = std::numeric_limits<T>::max();
  } else if (operation == static_cast<std::uint32_t>(kernel::ReduceOp::Max)) {
    value = std::numeric_limits<T>::lowest();
  }
  std::memcpy(output, &value, sizeof(value));
}

[[nodiscard]] bool reduce_identity(const Type type,
                                   const std::uint32_t operation,
                                   std::byte *const output) noexcept {
  if (operation != static_cast<std::uint32_t>(kernel::ReduceOp::Sum) &&
      operation != static_cast<std::uint32_t>(kernel::ReduceOp::CountNonzero) &&
      operation != static_cast<std::uint32_t>(kernel::ReduceOp::Min) &&
      operation != static_cast<std::uint32_t>(kernel::ReduceOp::Max)) {
    return false;
  }
  switch (type) {
  case Type::I32:
  case Type::FixedLane32:
    encode_reduce_identity<std::int32_t>(operation, output);
    return true;
  case Type::U32:
    encode_reduce_identity<std::uint32_t>(operation, output);
    return true;
  case Type::I64:
  case Type::FixedLane64:
    encode_reduce_identity<std::int64_t>(operation, output);
    return true;
  case Type::U64:
    encode_reduce_identity<std::uint64_t>(operation, output);
    return true;
  }
  return false;
}

} // namespace

using ::rund::detail::counter::Accumulate;

VirtualInputMaterializationResult materialize_virtual_input(
    VirtualBacking &backing, const VirtualRunProjection &run,
    const std::span<const VirtualInputMaterialization> pages,
    const VirtualInputReuseSeed *const prior) noexcept {
  VirtualInputMaterializationResult result{};
  if (pages.size() > PipelineLeafCapacity ||
      backing.size_bytes() != run.input_capacity_bytes) {
    result.status = Status::fail(Reason::PipelineInvalid);
    return result;
  }
  std::array<VirtualInputPageProjection, PipelineLeafCapacity> projected{};
  VirtualInputPageProjection prior_projection{};
  const bool has_prior =
      prior != nullptr && prior->frame != nullptr && !pages.empty() &&
      prior->page != std::numeric_limits<std::uint64_t>::max() &&
      prior->page + 1u == pages.front().page &&
      prior->frame != pages.front().frame &&
      project_virtual_input_page(run, prior->page, prior_projection);
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    if ((index != 0u && pages[index].page <= pages[index - 1u].page) ||
        ((pages[index].read || pages[index].finalize) &&
         pages[index].frame == nullptr) ||
        !project_virtual_input_page(run, pages[index].page, projected[index])) {
      result.status = Status::fail(Reason::PipelineInvalid);
      return result;
    }
  }
  const bool fill = run.clamp_window || run.clip_window || run.reduction();
  const std::size_t element_bytes =
      run.input_frame_elements == 0u
          ? 0u
          : static_cast<std::size_t>(run.input_page_bytes /
                                     run.input_frame_elements);
  std::array<std::byte, sizeof(std::uint64_t)> identity{};
  if (fill &&
      (element_bytes == 0u || element_bytes > identity.size() ||
       (run.clip_window &&
        !window_identity(run.input_type, run.operation, identity.data())) ||
       (run.reduction() &&
        !reduce_identity(run.input_type, run.operation, identity.data())))) {
    result.status = Status::fail(Reason::PrimitiveUnsupported);
    return result;
  }
  for (std::size_t index = 0u; index < pages.size(); ++index) {
    const VirtualInputMaterialization page = pages[index];
    const VirtualInputPageProjection projected_page = projected[index];
    if (page.read) {
      std::memset(page.frame, 0,
                  static_cast<std::size_t>(run.input_page_bytes));
      VirtualRead read{
          .offset = projected_page.logical_offset,
          .bytes =
              std::span<std::byte>{page.frame + projected_page.target_offset,
                                   projected_page.transfer_bytes},
      };
      VirtualInputReuseProjection reuse{};
      const std::byte *reuse_frame = nullptr;
      const VirtualInputPageProjection *reuse_projection = nullptr;
      if (index == 0u && has_prior) {
        reuse_frame = prior->frame;
        reuse_projection = &prior_projection;
      } else if (index != 0u && pages[index - 1u].frame != nullptr &&
                 pages[index - 1u].page !=
                     std::numeric_limits<std::uint64_t>::max() &&
                 pages[index - 1u].page + 1u == page.page &&
                 pages[index - 1u].frame != page.frame) {
        reuse_frame = pages[index - 1u].frame;
        reuse_projection = &projected[index - 1u];
      }
      if (reuse_projection != nullptr &&
          project_virtual_input_reuse(*reuse_projection, projected_page,
                                      reuse)) {
        std::memmove(page.frame + reuse.target_offset,
                     reuse_frame + reuse.source_offset, reuse.bytes);
        read = VirtualRead{
            .offset = reuse.logical_offset,
            .bytes = std::span<std::byte>{page.frame + reuse.read_target_offset,
                                          reuse.read_bytes},
        };
      }
      if (!read.bytes.empty()) {
        result.status = backing.read(read.offset, read.bytes);
        if (!result.status) {
          return result;
        }
        Accumulate(result.backing_bytes, read.bytes.size());
      }
      ++result.read_pages;
    }
    if (!page.finalize || !fill) {
      continue;
    }
    for (std::size_t offset = 0u; offset < projected_page.leading_fill_bytes;
         offset += element_bytes) {
      std::memcpy(page.frame + offset,
                  run.clamp_window
                      ? page.frame + projected_page.leading_fill_bytes
                      : identity.data(),
                  element_bytes);
    }
    if (projected_page.trailing_fill_offset < run.input_page_bytes) {
      const std::byte *const last =
          page.frame + projected_page.trailing_fill_offset - element_bytes;
      for (std::size_t offset = projected_page.trailing_fill_offset;
           offset < run.input_page_bytes; offset += element_bytes) {
        std::memcpy(page.frame + offset,
                    run.clamp_window ? last : identity.data(), element_bytes);
      }
    }
  }
  return result;
}

} // namespace rund::compute::detail
