#include "run.hpp"

#include "backend.hpp"
#include "cache.hpp"
#include "capacity.hpp"
#include "claim.hpp"
#include "failure.hpp"
#include "manifest.hpp"
#include "map.hpp"
#include "numeric.hpp"
#include "phase.hpp"
#include "publication.hpp"
#include "recipe.hpp"
#include "route.hpp"
#include "window.hpp"

namespace node_accel_contract {

bool AuthorityContract() {
  return BackendWindowDefaultsFailClosed() &&
         BackendWindowOccurrenceShapeHasOneAuthority() &&
         SubmissionTransitions() && PreparedPipelineClaimHasOneAuthority() &&
         PhaseSourceIdentityContract() &&
         PreparedPipelineFailureCoordinatesAreExact() &&
         PreparedPipelinePreparePathsAlwaysReportFailure() &&
         PreparedTemplateRegistryIsColdAndCollisionSafe() &&
         PreparedReservationIsFieldwiseFailClosed() &&
         RecurrenceRouteCopiesDoNotCloneTemplates() &&
         MetalPointerIdentityIndexIsExactAndOneShot() &&
         MetalIcbSizeClassPlanIsExactAndPortable() &&
         MetalColdIdentityIndexBudgetIsExact() &&
         MetalCaptureRowCapacityIsExact() &&
         PreparedTemplateStepCapacityIsBackendBounded() &&
         PreparedBackendProjectionIsRouteWiseAndChecked() &&
         PreparedPublicationCommandShapeIsCanonical() &&
         VulkanPhysicalCommandShapeIsNonOverlapping() &&
         VulkanPhysicalCapacityFailureIsTransactional() &&
         PreparedBackendControlManifestIsDimensionallyClosed() &&
         BackendSourceRecipeIsCheckedAndCanonical() &&
         VulkanMapAndResetSourceRecipesAreExact() &&
         MetalMapWordClassPartitionsProgramTemplates() &&
         MapSourceSpecializationIsSingleOwnerAndExact() &&
         MetalSourceRecipesAreExactAndSemantic() &&
         MetalMapCheckSourceHasOneGuardAuthority() &&
         MetalColdManifestIsExact() &&
         BackendTemplateRouteDemandIsExactAndFrozen() &&
         PreparedPublicationFingerprintIsSemantic() &&
         PreparedPublicationResolvedIdentityIsExact() && GridBoundaries() &&
         FinishPrecedence() && TelemetryProjection() &&
         BackendCapacityExceptionClassesAreCanonical();
}

} // namespace node_accel_contract
