#include "model.hpp"

namespace rund::node::accel::detail::device_vsm_graph_map_reduce {

void append_wavefront_candidates(std::string &source,
                                 const DeviceVsmGraphWavefrontProof &wavefront,
                                 const char *const and_token) {
  for (std::uint32_t stage = 0u; stage < wavefront.stage_count; ++stage) {
    const std::uint32_t bit = std::uint32_t{1u} << stage;
    const std::uint32_t same =
        wavefront.same_dispatch[stage] | wavefront.same_release[stage];
    const std::uint32_t prior =
        wavefront.prior_dispatch[stage] | wavefront.prior_release[stage];
    source += "      if (selected == ";
    source += std::to_string(wavefront.stage_count);
    source += "u ";
    source += and_token;
    source += " (completed & ";
    source += std::to_string(bit);
    source += "u) == 0u ";
    source += and_token;
    source += " (";
    source += std::to_string(same);
    source += "u & ~completed) == 0u ";
    source += and_token;
    source += " (batch == 0u || (";
    source += std::to_string(prior);
    source += "u & ~previous) == 0u)) selected = ";
    source += std::to_string(stage);
    source += "u;\n";
  }
}

} // namespace rund::node::accel::detail::device_vsm_graph_map_reduce
