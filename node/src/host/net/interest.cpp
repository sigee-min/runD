#include "interest.hpp"

#include "../../runtime/reactor/readiness/mask.hpp"

namespace rund::net {

node::ReactorInterest
ReactorInterestFor(const ready::Interest interest) noexcept {
  switch (interest) {
  case ready::Interest::Readable:
    return node::ReactorInterest::Read;
  case ready::Interest::Writable:
    return node::ReactorInterest::Write;
  case ready::Interest::ReadWrite:
    return node::ReactorInterest::Read | node::ReactorInterest::Write;
  }
  return node::ReactorInterest::None;
}

bool InterestFromReactor(const node::ReactorInterest interest,
                         ready::Interest *const out) noexcept {
  if (out == nullptr)
    return false;
  if (interest == node::ReactorInterest::Read) {
    *out = ready::Interest::Readable;
    return true;
  }
  if (interest == node::ReactorInterest::Write) {
    *out = ready::Interest::Writable;
    return true;
  }
  if (interest ==
      (node::ReactorInterest::Read | node::ReactorInterest::Write)) {
    *out = ready::Interest::ReadWrite;
    return true;
  }
  return false;
}

} // namespace rund::net
