#pragma once

#include "contract/program/compute/dsl/local.hpp"
#include "test/assert.hpp"

#include <string_view>

namespace program_compute_contract {

int RunComputeDslScalarOpsContract();
int RunComputeDslBitOpsContract();
int RunComputeDslOpsHashContract();
int RunComputeDslOpsNoiseContract();
int RunComputeDslOpsNoiseGridContract();
int RunComputeDslOpsNormContract();
int RunComputeDslOpsVecContract();
int RunComputeDslOpsSqContract();
int RunComputeDslOpsMetricContract();
int RunComputeDslOpsRangeContract();
int RunComputeDslOpsAggregateContract();
int RunComputeDslOpsStatsContract();
int RunComputeDslOpsMomentContract();
int RunComputeDslOpsCorrContract();
int RunComputeDslOpsRatioContract();
int RunComputeDslOpsStandardizeContract();
int RunComputeDslOpsApproxContract();
int RunComputeDslOpsTranscendentalContract();
int RunComputeDslOpsComplexContract();
int RunComputeDslOpsWindowSignalContract();
int RunComputeDslOpsLinearContract();
int RunComputeDslOpsMatrixContract();
int RunComputeDslOpsAffineContract();
int RunComputeDslOpsMixContract();
int RunComputeDslOpsPolyContract();
int RunComputeDslOpsInterpContract();
int RunComputeDslOpsMaskContract();
int RunComputeDslOpsTolContract();
int RunComputeDslOpsPieceContract();
int RunComputeDslOpsProjContract();
int RunComputeDslOpsReflectContract();
int RunComputeDslOpsCrossContract();
int RunComputeDslOpsLineContract();
int RunComputeDslOpsPlaneContract();
int RunComputeDslNonlinearOpsContract();
int RunComputeDslOpsOverloadContract();

} // namespace program_compute_contract
