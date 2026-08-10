#include "backing.hpp"

#include <memory>

namespace rund::compute {

VirtualBacking::VirtualBacking()
    : state_(std::make_unique<detail::VirtualBackingState>()) {}

VirtualBacking::~VirtualBacking() = default;

} // namespace rund::compute
