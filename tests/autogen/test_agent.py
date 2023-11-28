from unittest.mock import patch

import pytest
from autogen.agentchat import AssistantAgent
from autogen.oai.client import OpenAIWrapper

from fastagents.autogen.agent import (
    AutogenAgent,
    AzureLLMConfig,
    AzureOpenAIAPIVersions,
    OpenAILLMConfig,
    OpenAIModel,
)


class TestAzureOpenAIAPIVersions:
    def test___init__(self) -> None:
        azure_open_ai_api_versions = AzureOpenAIAPIVersions.v2023_12_01_preview
        assert isinstance(azure_open_ai_api_versions, str)
        assert azure_open_ai_api_versions is not None


class TestAzureLLMConfig:
    def test___init__bad_api_version(self) -> None:
        with pytest.raises(ValueError) as e:
            AzureLLMConfig(
                model="canada-gpt4",
                base_url="https://my-openai-canada.openai.azure.com",
                api_key="my-api-key",  # pragma: allowlist secret
                api_version="non-existing-api-version",
            )
        assert "validation error for AzureLLMConfig" in str(e.value), str(e.value)

    def test___init__(self) -> None:
        with patch("autogen.oai.client.OpenAI.__init__") as mock:
            mock.return_value = None
            config = AzureLLMConfig(
                model="canada-gpt4",
                base_url="https://my-openai-canada.openai.azure.com",
                api_key="my-api-key",  # pragma: allowlist secret
                api_version="2023-12-01-preview",
            )

            assert config is not None

            OpenAIWrapper(config_list=[config.model_dump()])
            mock.assert_called_once_with(
                base_url="https://my-openai-canada.openai.azure.com/openai/deployments/canada-gpt4",
                api_key="my-api-key",  # pragma: allowlist secret
                default_query={"api-version": "2023-12-01-preview"},
                default_headers={"api-key": "my-api-key"},
            )


class TestAzureOpenAIConfig:
    def test___init__(self) -> None:
        with patch("autogen.oai.client.OpenAI.__init__") as mock:
            mock.return_value = None
            config = OpenAILLMConfig(
                model=OpenAIModel.gpt_4_1106_preview,
                api_key="my-api-key",  # pragma: allowlist secret
            )

            assert config is not None

            OpenAIWrapper(config_list=[config.model_dump()])
            mock.assert_called_once_with(
                api_key="my-api-key",  # pragma: allowlist secret
            )


class TestAutogenAgent:
    def test___init__(self) -> None:
        agent = AutogenAgent(AssistantAgent)
        assert agent._functions == []
        assert agent._system_message == ""

    def test_function(self) -> None:
        agent = AutogenAgent(AssistantAgent)

        @agent.function
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers together

            Args:
                a (float): first number
                b (float): second number
            """
            return a + b

        @agent.function
        def multiply_numbers(a: float, b: float) -> float:
            """Multiply two numbers together

            Args:
                a (float): first number
                b (float): second number
            """
            return a * b

        assert set(agent._functions) == {add_numbers, multiply_numbers}
