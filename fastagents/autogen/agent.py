from typing import Any, Callable, Generic, List, Optional, Type, TypeVar, Union

from autogen.agentchat import Agent
from pydantic import BaseModel, Field, HttpUrl
from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated, Literal

from ..utils import Functions, parse_functions


def _check_base_url_end(base_url: str) -> str:
    # validate it is a valid URL
    HttpUrl(base_url)

    # make sure it does not end with a slash
    if base_url.endswith("/"):
        raise ValueError("Base URL must not end with a slash")

    # return unchanged
    return base_url


BaseUrl = Annotated[str, AfterValidator(_check_base_url_end)]


class AzureLLMConfig(BaseModel):
    """Azure OpenAI LLM Configuration"""

    model: str = Field(
        ...,
        description="Deployment name of the model",
        examples=["canada-gpt4", "sweden-gpt3.5"],
    )
    base_url: BaseUrl = Field(
        ...,
        description="Base URL of the Azure OpenAI service",
        examples=[
            "https://my-openai-canada.openai.azure.com",
            "https://my-openai-sweden.openai.azure.com",
        ],
    )
    api_key: str = Field(..., description="API key for the Azure OpenAI service")
    api_version: Literal[
        "2023-12-01-preview",
        "2023-10-01-preview",
        "2023-09-01-preview",
        "2023-08-01-preview",
        "2023-07-01-preview",
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


class LLMConfig(BaseModel):
    functions: Functions = Field(..., description="Functions for the LLM")
    config_list: List[Union[AzureLLMConfig, OpenAILLMConfig]] = Field(
        ..., description="List of LLM configurations"
    )
    timeout: int = Field(60, description="Timeout for the LLM")


AgentT = TypeVar("AgentT", bound=Agent)
AgentU = TypeVar("AgentU", bound=Agent)
CallableT = TypeVar("CallableT", bound=Callable[..., Any])


class AutogenAgent(Generic[AgentT]):
    def __init__(
        self,
        agent_cls: Type[AgentT],
        *args: Any,
        config_list: Optional[List[Union[AzureLLMConfig, OpenAILLMConfig]]] = None,
        **kwargs: Any,
    ) -> None:
        self._agent_cls: Type[AgentT] = agent_cls
        self._args = args
        self._kwargs = kwargs
        self._config_list: List[Union[AzureLLMConfig, OpenAILLMConfig]] = (
            [] if config_list is None else config_list
        )
        self._agent: Optional[AgentT] = None
        self._functions: List[Callable[..., Any]] = []

    def function(self, func: CallableT) -> CallableT:
        """Register a function with the agent

        Args:
            func(CallableT) : function to register

        Returns:
            function: function that was registered

        Raises:
            ValueError: if a function with the same name is already registered
        """
        if func.__name__ in [f.__name__ for f in self._functions]:
            raise ValueError(f"Function with name {func.__name__} already registered.")
        self._functions.append(func)

        return func

    def _create_llm_config(self) -> LLMConfig:
        functions = parse_functions(self._functions)
        return LLMConfig(
            functions=functions,
            config_list=self._config_list,
            timeout=self._kwargs.get("timeout", 60),
        )

    def _create_agent(self) -> None:
        # if self._functions != []:
        agent = self._agent_cls(
            *self._args,
            llm_config=self._create_llm_config().model_dump(),
            **self._kwargs,
        )
        agent.register_function({f.__name__: f for f in self._functions})

        self._agent = agent

    def initiate_chat(
        self, agent: "AutogenAgent[Any]", *args: Any, message: str, **kwargs: Any
    ) -> Any:
        self._create_agent()
        agent._create_agent()
        return self._agent.initiate_chat(agent._agent, *args, message=message, **kwargs)  # type: ignore[union-attr]


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
