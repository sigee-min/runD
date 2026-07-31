#include "local.hpp"

namespace rund::node::accel::detail {

BindingSource BindingSourceFor(const RunBinds& run_binds) noexcept {
  return BindingSource{
      .refs = run_binds.refs(),
      .handles = run_binds.handles(),
      .size = run_binds.size(),
  };
}

bool BindingSourceValid(const BindingSource& source) noexcept {
  return source.refs != nullptr && source.handles != nullptr;
}

bool ReadBinding(const BindingSource& source,
                 const std::uint64_t index,
                 const rund::kernel::ResidentBufferRef *&ref,
                 const std::shared_ptr<void> *&handle) noexcept {
  if (!BindingSourceValid(source) || index >= source.size) {
    return false;
  }
  const std::size_t local = static_cast<std::size_t>(index);
  ref = &source.refs[local];
  handle = &source.handles[local];
  return true;
}

bool ReadBindingPair(
    const BindingSource &source, const std::uint64_t first_index,
    const rund::kernel::ResidentBufferRef *&first,
    const std::shared_ptr<void> *&first_handle,
    const std::uint64_t second_index,
    const rund::kernel::ResidentBufferRef *&second,
    const std::shared_ptr<void> *&second_handle) noexcept {
  return ReadBinding(source, first_index, first, first_handle) &&
         ReadBinding(source, second_index, second, second_handle);
}

}  // namespace rund::node::accel::detail
