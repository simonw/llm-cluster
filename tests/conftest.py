import llm
from llm.plugins import pm
import pytest


class LengthSummary(llm.Model):
    model_id = "length-summary"

    def execute(self, prompt, stream, response, conversation):
        return ["Length: {}".format(len(prompt.prompt))]


class SimpleEmbeddings(llm.EmbeddingModel):
    model_id = "simple-embeddings"

    def embed_batch(self, texts):
        for text in texts:
            words = text.split()[:16]
            embedding = [len(word) for word in words]
            # Pad with 0 up to 16 words
            embedding += [0] * (16 - len(embedding))
            yield embedding


@pytest.fixture(autouse=True)
def register_models():
    class ModelsPlugin:
        __name__ = "ModelsPlugin"

        @llm.hookimpl
        def register_models(self, register):
            register(LengthSummary())

        @llm.hookimpl
        def register_embedding_models(self, register):
            register(SimpleEmbeddings())

    pm.register(ModelsPlugin(), name="undo-demo-plugin")
    try:
        yield
    finally:
        pm.unregister(name="undo-demo-plugin")
