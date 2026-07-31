#include "address.hpp"

#include "../hash/fields.hpp"

namespace rund::net {

::rund::StableHash HashAddress(const Address address) noexcept {
  if (!address) {
    return {};
  }

  // Address identity is exactly the semantic tuple
  // (family, 16 canonical address octets, host-order port). IPv4 occupies the
  // first four octets and the remaining twelve are fixed zeroes.
  constexpr std::uint64_t kCanonicalBytes = 1u + 16u + 2u;
  node::host_detail::CanonicalByteHasher hash(kCanonicalBytes);
  hash.AppendByte(static_cast<std::byte>(address.family()));
  hash.AppendBytes(address.bytes());
  if (address.family() == Family::IPv4) {
    constexpr std::array<std::byte, 12u> padding{};
    hash.AppendBytes(padding);
  }
  hash.AppendByte(static_cast<std::byte>(address.port() & 0xffu));
  hash.AppendByte(static_cast<std::byte>((address.port() >> 8u) & 0xffu));
  return hash.Finish();
}

} // namespace rund::net
