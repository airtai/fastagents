from typing import Any, Callable, Dict, get_type_hints

import docstring_parser
from pydantic import BaseModel, Field
from typing_extensions import Literal


class Parameter(BaseModel):
    type: str = Field(
        ..., description="Type of the parameter", examples=["float", "int", "str"]
    )
    description: str = Field(..., description="Description of the parameter")


class Parameters(BaseModel):
    type: Literal["object"] = Field("object")
    properties: Dict[str, Parameter]


class Function(BaseModel):
    description: str = Field(..., description="Description of the function")
    name: str = Field(..., description="Name of the function")
    parameters: Parameters = Field(..., description="Parameters of the function")


def parse_function(f: Callable[..., Any]) -> Function:
    """Parse a function into a Function object"""
    type_hints = get_type_hints(f)
    # check that we have all type hints for non-default variables
    parsed_docstring = docstring_parser.parse(f.__doc__ if f.__doc__ else "")

    descriptions = {p.arg_name: p.description for p in parsed_docstring.params}

    parameters = Parameters(
        properties={
            k: Parameter(
                type=v.__name__,
                description=(descriptions[k] if k in descriptions else k),
            )
            for k, v in type_hints.items()
            if k != "return"
        }
    )

    f_description = (
        parsed_docstring.short_description
        if parsed_docstring and parsed_docstring.short_description is not None
        else f.__name__
    )
    function = Function(
        description=f_description,
        name=f.__name__,
        parameters=parameters,
    )

    return function
