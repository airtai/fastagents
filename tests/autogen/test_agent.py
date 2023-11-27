from fastagents.autogen.agent import AutogenAgent


class TestAutogenAgent:
    def test___init__(self) -> None:
        agent = AutogenAgent()
        assert agent is not None
