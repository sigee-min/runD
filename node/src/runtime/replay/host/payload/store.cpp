#include "store.hpp"

#include "backend.hpp"
#include "store/local.hpp"

#include <stdexcept>
#include <utility>

namespace rund::node::replay_detail::payload {

Store::Store() : backend_{std::make_unique<Backend>()} {}

Store::Store(::rund::replay::Storage storage, const Limits limits,
             ::rund::replay::Diagnostic diagnostic)
    : storage_{std::move(storage)}, limits_{limits},
      staged_index_{limits.staged},
      backend_{std::make_unique<Backend>(storage_, limits.blobs)},
      diagnostic_{diagnostic} {
  if (limits_.bytes > storage_.max_bytes ||
      !store_detail::fits_u32(limits_.hosts + limits_.inputs) ||
      !store_detail::fits_u32(limits_.pieces) ||
      !store_detail::fits_u32(limits_.blobs) ||
      !store_detail::fits_u32(limits_.staged) ||
      limits_.staged > limits_.pieces || limits_.blobs > limits_.pieces) {
    throw std::invalid_argument{"replay_payload_store_limits_invalid"};
  }
  records_.reserve(limits_.hosts + limits_.inputs);
  host_record_indices_.reserve(limits_.hosts);
  input_record_indices_.reserve(limits_.inputs);
  pieces_.reserve(limits_.pieces);
  piece_scratch_.reserve(limits_.staged);
  staged_blobs_.reserve(limits_.staged);
}

Store::~Store() = default;

Store::Store(Store &&) noexcept = default;

Store &Store::operator=(Store &&) noexcept = default;

} // namespace rund::node::replay_detail::payload
