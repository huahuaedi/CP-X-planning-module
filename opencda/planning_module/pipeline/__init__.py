"""Apollo-style planning pipeline helpers.

This package keeps the high-level planning stages explicit:
cooperative perception input, prediction, behavior decision, and trajectory generation.
"""

from .behavior.candidate_evaluation import (
    BehaviorCandidate,
    CandidateEvaluationFrame,
    evaluate_behavior_candidates,
)
from .interaction.cav_interaction_stage import CAVInteractionStage
from .behavior.candidate_selection_stage import (
    CandidateArbitrationRequest,
    CandidateArbitrationResult,
    CandidateSelectionStage,
)
from .execution.actuator_mapper import ActuatorCommand, CarlaActuatorMapper
from .core.architecture_profile import ArchitectureProfile, normalize_architecture_config
from .behavior.candidate_pipeline import (
    CandidateBehaviorIntent,
    CandidateReferenceResult,
    CandidateSelectionOutcome,
    build_candidate_intents,
    evaluate_candidate_reference,
    select_best_candidate,
    select_candidate_with_commitment,
    summarize_candidate_results,
)
from .execution.control_buffer import MPCControlBuffer
from .behavior.behavior_decision import BehaviorConstraint, BehaviorDecision
from .behavior.behavior_stage import (
    BehaviorStage,
    BehaviorStageResult,
    BehaviorCommandFrameRequest,
    BehaviorCommandFrameResult,
    ConflictResolutionRequest,
    ConflictResolutionResult,
    RouteBehaviorContextResult,
)
from .safety.fallback_manager import (
    FailureReason,
    FallbackRequest,
    FallbackResult,
    TrajectoryFallbackManager,
)
from .core.decision_record import DecisionRecord, DecisionVeto, build_decision_record
from .behavior.destination_speed_stage import (
    DestinationSpeedStage,
    DestinationSpeedStageResult,
)
from .execution.execution_pipeline import (
    CooperativePlanningFrame,
    PlanningPipeline,
)
from .core.planning_context_stage import PlanningContextFrame, PlanningContextRequest
from .execution.mpc_feedback import BehaviorMPCFeedback
from .execution.mpc_command_extractor import MPCCommandExtractor, MPCTrackingCommand
from .execution.mpc_entry_stage import MPCEntryStage, MPCEntryStageResult
from .execution.mpc_cost_profile_stage import MPCCostProfileStage, MPCCostProfileState
from .execution.mpc_execution_stage import (
    MPCExecutionRequest,
    MPCExecutionResult,
    MPCExecutionStage,
)
from .behavior.maneuver_manager import LaneChangeLifecycle, ManeuverManager, TurnLifecycle
from .execution.nominal_trajectory import (
    NominalTarget,
    NominalTrajectory,
    NominalTrajectoryGenerator,
)
from .interaction.prediction import PredictionFrame, build_prediction_frame
from .core.output import BehaviorCommand, PlannerDiagnostics, PlannerOutput
from .perception.perception_stage import PerceptionStage, PerceptionStageResult
from .reference.reference_contract import (
    ReferenceContract,
    ReferenceValidationResult,
    contract_from_config,
    validate_reference_contract,
)
from .reference.reference_gate import FinalReferenceGate, FinalReferenceGateResult
from .reference.reference_line_provider import (
    ReferenceLineProvider,
    ReferenceLineRequest,
    ReferenceLineResult,
)
from .reference.reference_planning_stage import (
    BehaviorReferenceFrame,
    BehaviorReferencePreparationRequest,
    BehaviorReferenceRequest,
    CandidatePlanningPreparationRequest,
    PreparedBehaviorReference,
    PostTurnReferenceRequest,
    ReferencePlanningStage,
)
from .reference.reference_generator import (
    BoundaryRecoveryValidation,
    DrivableFootprintOccupancy,
    GeneratedReference,
    LaneCorridorOccupancy,
    ReferenceCorridorProjection,
    ReferenceGenerator,
)
from .reference.reference_pipeline import (
    ConditionedReference,
    ReferencePipeline,
    ReferencePipelineRequest,
    ReferencePipelineResult,
)
from .reference.reference_publication_stage import (
    ReferencePublicationStage,
    ReferencePublicationStageResult,
)
from .perception.runtime_input_stage import RuntimeInputStage, RuntimeTickSnapshot
from .behavior.traffic_light_memory import TrafficLightMemory
from .core.stage_contracts import ManeuverCommitment
from .route.route_authorization import (
    LaneChangeAuthorization,
    RouteManeuver,
    authorize_route_lane_change,
    normalize_route_maneuver,
)
from .route.route_manager import CPXRouteManager, RouteCursorSnapshot, RouteManagerStatus
from .safety.safety_supervisor import SafetySupervisor
from .behavior.scenario_manager import (
    BoundaryRecoveryRequest,
    CPXScenarioDecision,
    CPXScenarioManager,
)
from .behavior.speed_planner import (
    SpeedConstraint,
    SpeedPlan,
    SpeedTarget,
    SpeedTargetPlanner,
    build_speed_plan,
)
from .execution.velocity_steering_adapter import (
    CarlaVelocitySteeringAdapter,
    OpenCDAVelocitySteeringAdapter,
    VelocitySteeringCommand,
)
from .perception.tracker import CPXObstacleTracker

__all__ = [
    "ActuatorCommand",
    "CandidateArbitrationRequest",
    "CandidateArbitrationResult",
    "CandidateSelectionStage",
    "CAVInteractionStage",
    "BehaviorMPCFeedback",
    "BehaviorCommandFrameRequest",
    "BehaviorCommandFrameResult",
    "ConflictResolutionRequest",
    "ConflictResolutionResult",
    "RouteBehaviorContextResult",
    "PlanningContextFrame",
    "PlanningContextRequest",
    "MPCCommandExtractor",
    "MPCTrackingCommand",
    "MPCEntryStage",
    "MPCEntryStageResult",
    "MPCCostProfileStage",
    "MPCCostProfileState",
    "CarlaActuatorMapper",
    "BoundaryRecoveryRequest",
    "BoundaryRecoveryValidation",
    "DrivableFootprintOccupancy",
    "ArchitectureProfile",
    "BehaviorCandidate",
    "BehaviorCommand",
    "CandidateEvaluationFrame",
    "CandidateBehaviorIntent",
    "CandidateReferenceResult",
    "CandidateSelectionOutcome",
    "CPXRouteManager",
    "RouteCursorSnapshot",
    "CPXObstacleTracker",
    "CPXScenarioDecision",
    "CPXScenarioManager",
    "DecisionRecord",
    "DecisionVeto",
    "DestinationSpeedStage",
    "DestinationSpeedStageResult",
    "PlanningPipeline",
    "CooperativePlanningFrame",
    "ReferencePlanningStage",
    "BehaviorReferenceRequest",
    "BehaviorReferenceFrame",
    "BehaviorReferencePreparationRequest",
    "PreparedBehaviorReference",
    "CandidatePlanningPreparationRequest",
    "PostTurnReferenceRequest",
    "MPCControlBuffer",
    "BehaviorConstraint",
    "BehaviorDecision",
    "BehaviorStage",
    "BehaviorStageResult",
    "ManeuverManager",
    "LaneChangeLifecycle",
    "TurnLifecycle",
    "NominalTarget",
    "NominalTrajectory",
    "NominalTrajectoryGenerator",
    "FallbackResult",
    "FailureReason",
    "FallbackRequest",
    "TrajectoryFallbackManager",
    "PlannerDiagnostics",
    "PlannerOutput",
    "PerceptionStage",
    "PerceptionStageResult",
    "PredictionFrame",
    "ReferenceContract",
    "FinalReferenceGate",
    "FinalReferenceGateResult",
    "ReferenceGenerator",
    "ReferenceLineProvider",
    "ReferenceLineRequest",
    "ReferenceLineResult",
    "GeneratedReference",
    "LaneCorridorOccupancy",
    "ReferenceCorridorProjection",
    "ReferencePipeline",
    "ReferencePipelineRequest",
    "ReferencePipelineResult",
    "ReferencePublicationStage",
    "ReferencePublicationStageResult",
    "RuntimeInputStage",
    "RuntimeTickSnapshot",
    "ConditionedReference",
    "TrafficLightMemory",
    "ManeuverCommitment",
    "ReferenceValidationResult",
    "LaneChangeAuthorization",
    "RouteManagerStatus",
    "RouteManeuver",
    "SafetySupervisor",
    "SpeedPlan",
    "SpeedConstraint",
    "SpeedTarget",
    "SpeedTargetPlanner",
    "CarlaVelocitySteeringAdapter",
    "OpenCDAVelocitySteeringAdapter",
    "VelocitySteeringCommand",
    "authorize_route_lane_change",
    "build_prediction_frame",
    "build_candidate_intents",
    "build_decision_record",
    "build_speed_plan",
    "contract_from_config",
    "evaluate_behavior_candidates",
    "evaluate_candidate_reference",
    "normalize_route_maneuver",
    "normalize_architecture_config",
    "select_best_candidate",
    "select_candidate_with_commitment",
    "summarize_candidate_results",
    "validate_reference_contract",
]
