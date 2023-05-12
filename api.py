from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.responses import StreamingResponse

app = FastAPI()


class MptModelKwargs(BaseModel):
    instruction: str = "Hi, how are you?"
    temperature: float = 0.5
    top_p: float = 0.92
    top_k: int = 0
    max_new_tokens: int = 16384
    use_cache: bool = True
    do_sample: bool = True
    repetition_penalty: float = 1.1  # 1.0 means no pena
    # stop_tokens
    # presence_penalty
    # frequency_penalty


import os
from threading import Event, Thread
import torch
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from quick_pipeline import InstructionTextGenerationPipeline as pipeline

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN", None)


# Initialize the model and tokenizer
print("mosaicml/mpt-7b-chat")
generate_chat = pipeline(
    "mosaicml/mpt-7b-chat",
    # "gpt2",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_auth_token=HF_TOKEN,
)
stop_token_ids_chat = generate_chat.tokenizer.convert_tokens_to_ids(["<|endoftext|>"])


# Define a custom stopping criteria
class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in stop_token_ids_chat:
            if input_ids[0][-1] == stop_id:
                return True
        return False


def process_stream_chat(body: MptModelKwargs):
    # Tokenize the input
    input_ids = generate_chat.tokenizer(
        generate_chat.format_instruction(body.instruction), return_tensors="pt"
    ).input_ids
    input_ids = input_ids.to(generate_chat.model.device)

    # Initialize the streamer and stopping criteria
    streamer = TextIteratorStreamer(
        generate_chat.tokenizer,
        timeout=100.0,
        skip_prompt=True,
        skip_special_tokens=True,
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
        **generate_chat.generate_kwargs,
        **body2,
        **{
            "input_ids": input_ids,
            "streamer": streamer,
            "stopping_criteria": StoppingCriteriaList([stop]),
        },
    }

    stream_complete = Event()

    def generate_chat_and_signal_complete():
        generate_chat.model.generate(**gkw)
        stream_complete.set()

    def log_after_stream_complete():
        stream_complete.wait()

    t1 = Thread(target=generate_chat_and_signal_complete)
    t1.start()

    t2 = Thread(target=log_after_stream_complete)
    t2.start()

    for new_text in streamer:
        yield new_text


@app.post("/chat/")
async def create_chat(body: MptModelKwargs, stream=False):
    print(body.instruction)
    stream_chat = process_stream_chat(body)
    if stream:
        return StreamingResponse(stream_chat, media_type="text/event-stream")
    return StreamingResponse(stream_chat)


from InstructorEmbedding import INSTRUCTOR

model = INSTRUCTOR("hkunlp/instructor-large")
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"


@app.post("/instructor-embedding/")
def instructor_embedding(
    body: list[list[str, str]] = [[instruction, sentence]]
) -> list[list[float]]:
    e = model.encode(body)
    return e.tolist()


if __name__ == "__main__":
    uvicorn.run(app, port=8081, host="0.0.0.0")
