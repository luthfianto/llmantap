from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()


class ModelKwargs(BaseModel):
    instruction: str = "Hi, how are you?"
    temperature: float = 0.5
    top_p: float = 0.92
    top_k: int = 0
    max_new_tokens: int = 512
    use_cache: bool = True
    do_sample: bool = True
    repetition_penalty: float = 1.1  # 1.0 means no pena
    # stop_tokens
    # presence_penalty
    # frequency_penalty


app = FastAPI()


import os
from threading import Event, Thread
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from quick_pipeline import InstructionTextGenerationPipeline as pipeline

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", None)

# Initialize the model and tokenizer
generate = pipeline(
    "mosaicml/mpt-7b-instruct",
    # "gpt2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
stop_token_ids = generate.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])

# class MockIsiTo:
#     def to(self,_):
#         return None
# class MockIsiTokenizer:
#     input_ids=MockIsiTo()

# class MockTokenizer:
#     def __call__(self, *args: Any, **kwds: Any) -> Any:
#         return MockIsiTokenizer()
#     pad_token_id = None
#     pad_token = None
#     padding_side = "left"

# class MockModel():
#     device=None
#     def generate(self,**kwargs): return {}
# class MockGenerate:
#     def __init__(self) -> None:
#         self.tokenizer = MockTokenizer()
#         self.model=MockModel()
#     def format_instruction(self, _a): pass
#     generate_kwargs={}


# generate = MockGenerate()
# stop_token_ids=[]


# Define a custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def process_stream2(body: ModelKwargs):
    # Tokenize the input
    input_ids = generate.tokenizer(
        generate.format_instruction(body.instruction), return_tensors="pt"
    ).input_ids
    input_ids = input_ids.to(generate.model.device)

    # Initialize the streamer and stopping criteria
    streamer = TextIteratorStreamer(
        generate.tokenizer, timeout=100.0, skip_prompt=True, skip_special_tokens=True
    )
    stop = StopOnTokens()

    if body.temperature < 0.1:
        body.temperature = 0.0
        body.do_sample = False
    else:
        body.do_sample = True

    body2 = dict(body)
    del body2["instruction"]
    print(body2)

    gkw = {
        **generate.generate_kwargs,
        **body2,
        **{
            "input_ids": input_ids,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([stop]),
        },
    }

    stream_complete = Event()

    def generate_and_signal_complete():
        generate.model.generate(**gkw)
        stream_complete.set()

    def log_after_stream_complete():
        stream_complete.wait()

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    t2 = Thread(target=log_after_stream_complete)
    t2.start()

    for new_text in streamer:
        yield new_text


from fastapi.responses import StreamingResponse


@app.post("/llm/")
async def create_item(body: ModelKwargs):
    stream = process_stream2(body)
    return StreamingResponse(stream)


if __name__ == "__main__":
    uvicorn.run(app, port=8081, host="0.0.0.0")
