from typing import Any, Callable, Generic, List, Optional, Type, TypeVar

from autogen.agentchat import Agent
from pydantic import BaseModel, Field
from typing_extensions import Literal

from ..utils import parse_functions


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
    api_version: Literal[
        "2023-07-01-preview",
        "2023-08-01-preview",
        "2023-09-01-preview",
        "2023-10-01-preview",
        "2023-12-01-preview",
    ] = Field(
        description="API version for the Azure OpenAI service",
        examples=["2023-12-01-preview"],
        default="2023-12-01-preview",
    )

    api_type: Literal["azure"] = Field("azure")


class OpenAILLMConfig(BaseModel):
    """OpenAI LLM Configuration"""

    model: Literal[
        "gpt-4",
        "gpt-4-1106-preview",
        "gpt-4-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0613",
    ] = Field(
        ...,
        description="Deployment name of the model",
        examples=["gpt-4", "gpt-3.5-turbo-1106"],
    )
    api_key: str = Field(..., description="API key for the OpenAI service")


AgentT = TypeVar("AgentT", bound=Agent)
AgentU = TypeVar("AgentU", bound=Agent)
CallableT = TypeVar("CallableT", bound=Callable[..., Any])


class AutogenAgent(Generic[AgentT]):
    def __init__(self, agent_cls: Type[AgentT], *args: Any, **kwargs: Any) -> None:
        self._args = args
        self._kwargs = kwargs
        self._agent_cls: Type[AgentT] = agent_cls
        self._agent: Optional[AgentT] = None
        self._functions: List[Callable[..., Any]] = []
        self._system_message: Optional[str] = ""

    def function(self, func: CallableT) -> CallableT:
        self._functions.append(func)
        return func

    def _create_agent(self) -> None:
        functions = parse_functions(self._functions)

        self._agent = self._agent_cls(*self._args, **self._kwargs, functions=functions)

    def start_conversation(
        self, agent: "AutogenAgent[Any]", initial_message: str
    ) -> Any:
        self._create_agent()
        agent._create_agent()
        return self._agent.start_conversation(initial_message)  # type: ignore[union-attr]


# class AutogenTeam:
#     def __init__(self, *args: Any, **kwargs: Any) -> None:
#         self._args = args
#         self._kwargs = kwargs
#         self._group_chat: Optional[GroupChat] = None
#         self._agents: List[AutogenAgent[Any]] = []

#     def add_agent(self, agent: AutogenAgent[Any]) -> None:
#         self._agents.append(agent)

#     def create(self) -> GroupChat:
#         group_chat = GroupChat(*self._args, **self._kwargs)
#         self._group_chat = group_chat

#         return group_chat

#     def start_chat(self, initial_message: str) -> None:
#         self._group_chat.initiate_chat(initial_message)
