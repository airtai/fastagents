import pytest

from fastagents.utils.docstring import _parse_function


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together

    Args:
        a (float): first number
        b (float): second number
    """
    return a * b


def add_numbers(a: float, b: float) -> float:
    """Add two numbers together

    Args:
        a: first number
    """
    return a + b


def substract_numbers(a, b) -> float:  # type: ignore[no-untyped-def]
    return a - b  # type: ignore[no-any-return]


def test_parse_function() -> None:
    expected = """{
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
}"""
    actual = _parse_function(multiply_numbers).model_dump_json(indent=2)
    assert actual == expected, actual


def test_parse_uncomplete_function() -> None:
    expected = """{
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
        "description": "b"
      }
    }
  }
}"""
    actual = _parse_function(add_numbers).model_dump_json(indent=2)
    assert actual == expected, actual


@pytest.mark.skip(reason="Not implemented yet")
def test_parse_functions() -> None:
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_parse_barebone_function() -> None:
    expected = """{
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
        "description": "b"
      }
    }
  }
}"""
    actual = _parse_function(substract_numbers).model_dump_json(indent=2)
    assert actual == expected, actual
