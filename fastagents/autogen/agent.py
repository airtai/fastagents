from enum import Enum
from typing import Any, Callable, Generic, List, Optional, Type, TypeVar

from autogen.agentchat import Agent, GroupChat
from pydantic import BaseModel, Field
from typing_extensions import Literal


class AzureOpenAIAPIVersions(str, Enum):
    v2023_12_01_preview = "2023-12-01-preview"
    v2023_10_01_preview = "2023-10-01-preview"
    v2023_09_01_preview = "2023-09-01-preview"
    v2023_08_01_preview = "2023-08-01-preview"
    v2023_07_01_preview = "2023-07-01-preview"


class AzureLLMConfig(BaseModel):
    """Azure OpenAI LLM Configuration"""

    model: str = Field(
        ...,
        description="Deployment name of the model",
        examples=["canada-gpt4", "sweden-gpt3.5"],
    )
    base_url: str = Field(
        ...,
        description="Base URL of the Azure OpenAI service",
        examples=[
            "https://my-openai-canada.openai.azure.com",
            "https://my-openai-sweden.openai.azure.com",
        ],
    )
    api_key: str = Field(..., description="API key for the Azure OpenAI service")
    api_version: AzureOpenAIAPIVersions = Field(
        # ...,
        description="API version for the Azure OpenAI service",
        examples=["2023-12-01-preview"],
        default=AzureOpenAIAPIVersions.v2023_12_01_preview,
    )

    api_type: Literal["azure"] = Field("azure")


class OpenAIModel(str, Enum):
    gpt_4 = "gpt-4"
    gpt_4_1106_preview = "gpt-4-1106-preview"
    gpt_4_0613 = "gpt-4-0613"
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_3_5_turbo_1106 = "gpt-3.5-turbo-1106"
    gpt_3_5_turbo_0613 = "gpt-3.5-turbo-0613"


class OpenAILLMConfig(BaseModel):
    """OpenAI LLM Configuration"""

    model: OpenAIModel = Field(
        ...,
        description="Deployment name of the model",
        examples=["gpt-4", "gpt-3.5-turbo-1106"],
    )
    api_key: str = Field(..., description="API key for the OpenAI service")


AgentT = TypeVar("AgentT", bound=Agent)
CallableT = TypeVar("CallableT", bound=Callable[..., Any])


class AutogenAgent(Generic[AgentT]):
    def __init__(self, agent_cls: Type[AgentT]) -> None:
        self._agent_cls: Type[AgentT] = agent_cls
        self._agent: Optional[AgentT] = None
        self._functions: List[Callable[..., Any]] = []
        self._system_message: Optional[str] = ""

    def function(self, func: CallableT) -> CallableT:
        self._functions.append(func)
        return func

    def system_message(self, message: str) -> None:
        self._system_message = message

    def _create_agent(self) -> AgentT:
        agent: AgentT = self._agent_cls()
        self._agent = agent
        return agent


class AutogenTeam:
    def __init__(self) -> None:
        self._group_chat: GroupChat = GroupChat()
        self._agents: List[AutogenAgent[Any]] = []

    def add_agent(self, agent: AutogenAgent[Any]) -> None:
        self._agents.append(agent)

    def start_chat(self, initial_mesage: str) -> None:
        raise NotImplementedError

    async def a_start_chat(self, initial_mesage: str) -> None:
        raise NotImplementedError
