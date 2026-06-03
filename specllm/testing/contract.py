"""Contract testing framework."""

from dataclasses import dataclass, field
from typing import List

from specllm.spec.parser import Endpoint
from specllm.spec.validator import validate_schema, ValidationError
from specllm.prompts.generator import generate_prompt


@dataclass
class ContractTestResults:
    """Results from running contract tests."""

    total: int
    passed: int
    failed: int
    violations: List[ValidationError] = field(default_factory=list)


class ContractTestRunner:
    """Runs contract tests against an LLM provider."""

    def __init__(self, provider: object) -> None:
        self.provider = provider

    def run(self, endpoint: Endpoint, samples: List[dict]) -> ContractTestResults:
        """Run contract tests for an endpoint with sample inputs."""
        total = len(samples)
        passed = 0
        failed = 0
        violations: List[ValidationError] = []

        for sample in samples:
            prompt = generate_prompt(endpoint, sample)
            output = self.provider.call(prompt)

            if endpoint.response_schema:
                errors = validate_schema(output, endpoint.response_schema)
                if errors:
                    failed += 1
                    violations.extend(errors)
                else:
                    passed += 1
            else:
                passed += 1

        return ContractTestResults(total=total, passed=passed, failed=failed, violations=violations)
