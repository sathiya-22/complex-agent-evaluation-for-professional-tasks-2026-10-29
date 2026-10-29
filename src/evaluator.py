from typing import List, Dict, Any
from src.models import AgentTranscript, GroundTruth, TaskDefinition, EvaluationResult, OverallEvaluation, ExpectedToolCall
import json

class AgentEvaluator:
    def __init__(self, task_definition: TaskDefinition, ground_truth: GroundTruth):
        self.task_definition = task_definition
        self.ground_truth = ground_truth

    def _evaluate_tool_calls(self, agent_transcript: AgentTranscript) -> EvaluationResult:
        agent_tool_calls = [tc for step in agent_transcript.steps for tc in step.tool_calls]
        expected_tool_calls = self.ground_truth.expected_tool_calls_sequence

        correct_calls = 0
        total_expected_calls = len(expected_tool_calls)
        matched_expected_indices = set()

        # Simple sequential matching for now; could be more sophisticated (e.g., fuzzy matching, graph isomorphism)
        for i, expected_tc in enumerate(expected_tool_calls):
            # Find the first unmatched agent tool call that matches the expected one
            for j, agent_tc in enumerate(agent_tool_calls):
                if j not in matched_expected_indices and \
                   agent_tc.tool_name == expected_tc.tool_name and \
                   self._match_arguments(agent_tc.arguments, expected_tc.arguments):
                    correct_calls += 1
                    matched_expected_indices.add(j)
                    break # Move to the next expected tool call

        score = (correct_calls / total_expected_calls) * 100 if total_expected_calls > 0 else 100
        return EvaluationResult(
            metric="tool_call_accuracy",
            score=score,
            details={
                "correct_calls": correct_calls,
                "total_expected_calls": total_expected_calls,
                "agent_calls_made": len(agent_tool_calls),
                "unmatched_agent_calls": len(agent_tool_calls) - correct_calls # A proxy for hallucinated calls
            }
        )

    def _match_arguments(self, agent_args: Dict[str, Any], expected_args: Dict[str, Any]) -> bool:
        """
        Compares agent arguments to expected arguments.
        Supports exact match for values. For more complex cases, could use regex or semantic comparison.
        """
        for key, expected_value in expected_args.items():
            if key not in agent_args:
                return False
            agent_value = agent_args[key]
            if expected_value == "__ANY__": # Special placeholder for any value
                continue
            if agent_value != expected_value:
                return False
        return True

    def _evaluate_final_answer(self, agent_transcript: AgentTranscript) -> EvaluationResult:
        agent_answer = agent_transcript.final_answer
        expected_answer = self.ground_truth.expected_final_answer

        is_correct = False
        if isinstance(expected_answer, dict) and isinstance(agent_answer, dict):
            # For structured answers, check if all expected keys/values are present and match
            is_correct = True
            for key, expected_val in expected_answer.items():
                if key not in agent_answer:
                    is_correct = False
                    break
                if expected_val == "__ANY__":
                    continue
                if agent_answer[key] != expected_val:
                    is_correct = False
                    break
        elif isinstance(expected_answer, (int, float)) and isinstance(agent_answer, (int, float)):
            tolerance = self.task_definition.expected_final_output.tolerance
            if tolerance is not None:
                is_correct = abs(agent_answer - expected_answer) <= tolerance
            else:
                is_correct = (agent_answer == expected_answer)
        else:
            is_correct = (agent_answer == expected_answer)

        score = 100 if is_correct else 0
        return EvaluationResult(
            metric="final_answer_accuracy",
            score=score,
            details={
                "agent_answer": agent_answer,
                "expected_answer": expected_answer,
                "match": is_correct
            }
        )

    def evaluate(self, agent_transcript: AgentTranscript) -> OverallEvaluation:
        results: List[EvaluationResult] = []

        tool_call_eval = self._evaluate_tool_calls(agent_transcript)
        results.append(tool_call_eval)

        final_answer_eval = self._evaluate_final_answer(agent_transcript)
        results.append(final_answer_eval)

        # Calculate overall score (simple average for now)
        overall_score = sum(r.score for r in results) / len(results) if results else 0

        return OverallEvaluation(
            task_id=agent_transcript.task_id,
            overall_score=overall_score,
            results=results
        )
