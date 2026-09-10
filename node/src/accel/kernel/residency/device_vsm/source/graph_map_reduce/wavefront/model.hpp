#pragma once

#include "../../../proof.hpp"

#include <string>

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

void append_wavefront_candidates(std::string &,
                                 const DeviceVsmGraphWavefrontProof &,
                                 const char *and_token);

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
