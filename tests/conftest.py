import llm
from llm.plugins import pm
import pytest


class LengthSummary(llm.Model):
    model_id = "length-summary"

    def execute(self, prompt, stream, response, conversation):
        self.last_prompt = prompt
        return ["Length: {}".format(len(prompt.prompt))]


@pytest.fixture
def length_summary():
    return LengthSummary()


@pytest.fixture(autouse=True)
def register_embed_demo_model(length_summary):
    class LengthSummaryPlugin:
        __name__ = "LengthSummaryPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(length_summary)

    pm.register(LengthSummaryPlugin(), name="undo-demo-plugin")
    try:
        yield
    finally:
        pm.unregister(name="undo-demo-plugin")
