#include <kernel/program/compute/lowering/parse.hpp>

namespace rund::kernel::compute_lowering_detail {

Reader::Reader(const std::vector<u8> &bytes) noexcept : bytes_(&bytes) {}

bool Reader::read_u8(u8 &value) noexcept {
  if (remaining() < 1u)
    return false;
  value = (*bytes_)[offset_++];
  return true;
}

bool Reader::read_u32(u32 &value) noexcept {
  if (remaining() < 4u)
    return false;
  value = static_cast<u32>((*bytes_)[offset_]) |
          (static_cast<u32>((*bytes_)[offset_ + 1u]) << 8u) |
          (static_cast<u32>((*bytes_)[offset_ + 2u]) << 16u) |
          (static_cast<u32>((*bytes_)[offset_ + 3u]) << 24u);
  offset_ += 4u;
  return true;
}

bool Reader::read_bytes(std::vector<u8> &out) {
  u32 size = 0u;
  if (!read_u32(size) || remaining() < size)
    return false;
  out.assign(bytes_->begin() + static_cast<std::ptrdiff_t>(offset_),
             bytes_->begin() + static_cast<std::ptrdiff_t>(offset_ + size));
  offset_ += size;
  return true;
}

bool Reader::read_string(std::string &out) {
  u32 size = 0u;
  if (!read_u32(size) || remaining() < size)
    return false;
  out.assign(reinterpret_cast<const char *>(bytes_->data() + offset_), size);
  offset_ += size;
  return true;
}

bool Reader::done() const noexcept {
  return bytes_ != nullptr && offset_ == bytes_->size();
}

std::size_t Reader::remaining() const noexcept {
  return bytes_ == nullptr || offset_ > bytes_->size()
             ? 0u
             : bytes_->size() - offset_;
}

} // namespace rund::kernel::compute_lowering_detail
