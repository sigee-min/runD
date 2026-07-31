#include "hash.hpp"

#include "../../hash/bytes.hpp"
#include "algorithm.hpp"

namespace rund::net {
namespace {

class Identity final {
public:
  class Sequence final {
  public:
    void Append(const std::span<const std::byte> bytes) noexcept {
      hash_.Append(bytes);
    }

    [[nodiscard]] ::rund::StableHash Finish() const noexcept {
      return hash_.Finish();
    }

  private:
    node::host_detail::StableByteHasher hash_{};
  };

  [[nodiscard]] ::rund::StableHash Hash(const std::byte *const data,
                                        const std::size_t size) const noexcept {
    return node::host_detail::StableByteHash(data, size);
  }

  [[nodiscard]] Sequence Begin() const noexcept { return {}; }
};

} // namespace

::rund::StableHash
PayloadHashForNative(const node::NativeCallResult &native,
                     const std::byte *const data,
                     const std::uint64_t requested) noexcept {
  Identity identity{};
  return payload_detail::Native(native, data, requested, identity);
}

::rund::StableHash
PayloadHashForPrefix(const std::span<const batch::Slice> slices,
                     const std::uint64_t completed_bytes) noexcept {
  Identity identity{};
  return payload_detail::Prefix(slices, completed_bytes, identity);
}

} // namespace rund::net
