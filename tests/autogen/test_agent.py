import unittest
from unittest.mock import MagicMock, patch

import pytest
from autogen.agentchat import AssistantAgent
from autogen.oai.client import OpenAIWrapper
from pydantic import BaseModel

from fastagents.autogen import (
    AutogenAgent,
    AzureLLMConfig,
    BaseUrl,
    OpenAILLMConfig,
)
from fastagents.autogen.agent import LLMConfig
from fastagents.utils.docstring import Functions


class TestBaseUrl:
    class MyBaseUrl(BaseModel):
        base_url: BaseUrl

    def test_correct(self) -> None:
        my_base_url = TestBaseUrl.MyBaseUrl(
            base_url="https://my-openai-canada.openai.azure.com"
        )
        assert my_base_url.base_url == "https://my-openai-canada.openai.azure.com"

    def test_ending_with_slash(self) -> None:
        with pytest.raises(ValueError) as e:
            TestBaseUrl.MyBaseUrl(base_url="https://my-openai-canada.openai.azure.com/")
        assert "Base URL must not end with a slash" in str(e.value)


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
                model="gpt-4-1106-preview",
                api_key="my-api-key",  # pragma: allowlist secret
            )

            assert config is not None

            OpenAIWrapper(config_list=[config.model_dump()])
            mock.assert_called_once_with(
                api_key="my-api-key",  # pragma: allowlist secret
            )


class TestAutogenAgent:
    def test___init__(self) -> None:
        agent = AutogenAgent(
            AssistantAgent,
            system_message="Hello, I am an autogen agent",
            config_list=[
                OpenAILLMConfig(
                    model="gpt-4-1106-preview",
                    api_key="my-api-key",  # pragma: allowlist secret
                )
            ],
        )

        assert agent._agent_cls == AssistantAgent, agent._agent_cls
        assert agent._functions == []
        assert agent._kwargs["system_message"] == "Hello, I am an autogen agent"

    def test_function(self) -> None:
        agent = AutogenAgent(
            AssistantAgent,
            config_list=[
                OpenAILLMConfig(
                    model="gpt-4",
                    api_key="api-key",  # pragma: allowlist secret
                )
            ],
        )

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

    def test_create_llm_config(self) -> None:
        agent = AutogenAgent(
            AssistantAgent,
            config_list=[
                OpenAILLMConfig(
                    model="gpt-4",
                    api_key="api-key",  # pragma: allowlist secret
                )
            ],
        )

        def add_numbers(a: float, b: float) -> float:
            """Add two numbers together

            Args:
                a (float): first number
                b (float): second number
            """
            return a + b

        def multiply_numbers(a: float, b: float) -> float:
            """Multiply two numbers together

            Args:
                a (float): first number
                b (float): second number
            """
            return a * b

        agent._functions = [add_numbers, multiply_numbers]

        llm_config = agent._create_llm_config()

        assert isinstance(llm_config, LLMConfig)
        assert isinstance(llm_config.functions, Functions)

        # pragma: allowlist secret
        expected = """{
  "functions": {
    "description": "A list of functions the model may generate JSON inputs for.",
    "type": "array",
    "minItems": 1,
    "items": [
      {
        "type": "function",
        "function": {
          "description": "Add two numbers together",
          "name": "add_numbers",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {
                "type": "float",
                "description": "first number"
              },
              "b": {
                "type": "float",
                "description": "second number"
              }
            }
          }
        }
      },
      {
        "type": "function",
        "function": {
          "description": "Multiply two numbers together",
          "name": "multiply_numbers",
          "parameters": {
            "type": "object",
            "properties": {
              "a": {
                "type": "float",
                "description": "first number"
              },
              "b": {
                "type": "float",
                "description": "second number"
              }
            }
          }
        }
      }
    ]
  },
  "config_list": [
    {
      "model": "gpt-4",
      "api_key": "api-key"
    }
  ],
  "timeout": 60
}"""  # pragma: allowlist secret

        assert llm_config.model_dump_json(indent=2) == expected

    def test_create_agent(self) -> None:
        with unittest.mock.patch.object(
            AssistantAgent, "register_function", return_value=None
        ) as mock_register_function:
            agent = AutogenAgent(
                AssistantAgent,
                config_list=[
                    OpenAILLMConfig(
                        model="gpt-4", api_key="api-key"  # pragma: allowlist secret
                    )
                ],
            )

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

            agent._functions = [add_numbers, multiply_numbers]

            mock_agent_cls = agent._agent_cls = MagicMock(
                return_value=AssistantAgent(name="test")
            )
            agent._create_agent()

            # First, check that the mock was calleda
            mock_agent_cls.assert_called_once()

            # Next, check that the mock was called with the correct arguments
            expected_kwargs = {
                "functions": {
                    "description": "A list of functions the model may generate JSON inputs for.",
                    "type": "array",
                    "minItems": 1,
                    "items": [
                        {
                            "type": "function",
                            "function": {
                                "description": "Add two numbers together",
                                "name": "add_numbers",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "a": {
                                            "type": "float",
                                            "description": "first number",
                                        },
                                        "b": {
                                            "type": "float",
                                            "description": "second number",
                                        },
                                    },
                                },
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "description": "Multiply two numbers together",
                                "name": "multiply_numbers",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        "a": {
                                            "type": "float",
                                            "description": "first number",
                                        },
                                        "b": {
                                            "type": "float",
                                            "description": "second number",
                                        },
                                    },
                                },
                            },
                        },
                    ],
                }
            }
            print(mock_agent_cls.call_args_list)
            mock_agent_cls.assert_called_once_with(**expected_kwargs)

            # Finally, check that the register_function was called with the correct arguments
            mock_register_function.assert_called_once_with(
                {"add_numbers": add_numbers, "multiply_numbers": multiply_numbers}
            )
