"""Tests for specllm.testing.contract module - contract testing."""
import unittest
from unittest.mock import Mock, MagicMock

from specllm.testing.contract import ContractTestRunner, ContractTestResults
from specllm.spec.parser import Endpoint

class TestContractTestRunner(unittest.TestCase):
    """Test contract testing framework."""

    def setUp(self):
        """Set up contract test runner with mock provider."""
        self.endpoint = Endpoint(
            path="/users",
            method="post",
            description="Create a user",
            request_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["name", "email"],
            },
            response_schema={
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string"},
                    "email": {"type": "string"},
                },
                "required": ["id", "name", "email"],
            },
            parameters=[],
        )

        self.sample_inputs = [
            {"name": "Alice", "email": "alice@example.com"},
            {"name": "Bob", "email": "bob@example.com"},
            {"name": "Charlie", "email": "charlie@example.com"},
        ]

    def test_contract_passes_when_output_valid(self):
        """Contract test passes when mock LLM returns valid output."""
        mock_provider = Mock()
        mock_provider.call.return_value = {
            "id": 1,
            "name": "Alice",
            "email": "alice@example.com",
        }

        runner = ContractTestRunner(provider=mock_provider)
        results = runner.run(self.endpoint, self.sample_inputs[:1])

        self.assertIsInstance(results, ContractTestResults)
        self.assertEqual(results.passed, 1)
        self.assertEqual(results.failed, 0)

    def test_contract_fails_when_output_violates_schema(self):
        """Contract test fails when output violates schema."""
        mock_provider = Mock()
        mock_provider.call.return_value = {"invalid": "data"}

        runner = ContractTestRunner(provider=mock_provider)
        results = runner.run(self.endpoint, self.sample_inputs[:1])

        self.assertEqual(results.failed, 1)
        self.assertEqual(results.passed, 0)

    def test_runner_processes_multiple_samples(self):
        """Test runner processes multiple sample inputs."""
        mock_provider = Mock()
        mock_provider.call.return_value = {
            "id": 1,
            "name": "Test",
            "email": "test@example.com",
        }

        runner = ContractTestRunner(provider=mock_provider)
        results = runner.run(self.endpoint, self.sample_inputs)

        self.assertEqual(results.total, 3)
        self.assertEqual(mock_provider.call.call_count, 3)

    def test_results_include_counts(self):
        """Results include total, passed, failed counts."""
        mock_provider = Mock()
        mock_provider.call.side_effect = [
            {"id": 1, "name": "Alice", "email": "a@b.com"},
            {"bad": "output"},  # fails
            {"id": 3, "name": "Charlie", "email": "c@d.com"},
        ]

        runner = ContractTestRunner(provider=mock_provider)
        results = runner.run(self.endpoint, self.sample_inputs)

        self.assertEqual(results.total, 3)
        self.assertEqual(results.passed, 2)
        self.assertEqual(results.failed, 1)

    def test_results_include_violations(self):
        """Results include violation details for failures."""
        mock_provider = Mock()
        mock_provider.call.return_value = {"bad": "output"}

        runner = ContractTestRunner(provider=mock_provider)
        results = runner.run(self.endpoint, self.sample_inputs[:1])

        self.assertIsInstance(results.violations, list)
        self.assertGreater(len(results.violations), 0)

if __name__ == "__main__":
    unittest.main()
