from __future__ import annotations

import pymongo, grpc, time, asyncio, warnings

from typing import Any, AsyncIterator, Dict, List, cast, Mapping, Optional, Iterator, Type

from agent_pb2 import TextRequest, Query, Docs, Compress, MessageRequest
from agent_pb2_grpc import llmStub, embedingStub, compressed_promptStub

from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, get_buffer_string
from langchain_core.pydantic_v1 import root_validator, BaseModel, Extra, Field
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.outputs import GenerationChunk, LLMResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain.chains import LLMChain
from langchain.memory.utils import get_prompt_input_key
from langchain.memory.chat_memory import BaseChatMemory
from langchain.callbacks.base import AsyncCallbackHandler



warnings.filterwarnings('ignore')

class LLM(LLM):
    host: str = 'localhost:50051'
    pipeline: Any
    model_kwargs: Optional[dict] = None
    pipeline_kwargs: Optional[dict] = None
    batch_size: int = 4
    stream_out: bool = False
    stub: Any = None
    uid: str = None

    @property
    def _llm_type(self) -> str:
        return "TA"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        result = ""
        for response in self._stream(prompt, stop, run_manager, **kwargs):
            if self.stream_out:
                print(response.text, end="", flush=True)
            else:
                response.text
            result += response.text
        return result
    
    def _stream(self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        if self.uid == None or self.uid == "":
            raise Exception("uid is required")
        if self.stub == None:
            self.stub = llmStub(grpc.insecure_channel(self.host))
        responses = self.stub.stream(TextRequest(prompt=prompt, model_kwargs=str(self.model_kwargs), pipeline_kwargs=str(self.pipeline_kwargs), id=self.uid))
        for response in responses:
            chunk = GenerationChunk(text=response.result)
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    def get_num_tokens_from_messages(self, message: str):
        if self.uid == None or self.uid == "":
            raise Exception("uid is required")
        if self.stub == None:
            self.stub = llmStub(grpc.insecure_channel(self.host))
        return self.stub.get_num_tokens_from_messages(MessageRequest(prompt = message)).num


    def get_tokens_from_messages(self, message: str):
        if self.uid == None or self.uid == "":
            raise Exception("uid is required")
        if self.stub == None:
            self.stub = llmStub(grpc.insecure_channel(self.host))
        return self.stub.get_tokens_from_messages(MessageRequest(prompt = message)).token
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {'model_kwargs': self.model_kwargs, 'pipeline_kwargs': self.pipeline_kwargs, 'stream': self.stream_out}
    
    

class Embedding(BaseModel, Embeddings):
    
    client: Any
    cache_folder: Optional[str] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    embed_instruction: str = "Represent the document for retrieval: "
    query_instruction: str = ("Represent the question for retrieving supporting documents: ")
    host: str = 'localhost:50051'    
    stub: Any = None
        
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if self.stub == None:
            self.stub = embedingStub(grpc.insecure_channel(self.host))
        result = self.stub.embed_documents(Docs(docs = texts, embed_inst=self.embed_instruction)).docs
        return [doc.query for doc in result]
        
    def embed_query(self, text: str) -> List[float]:
        if self.stub == None:
            self.stub = embedingStub(grpc.insecure_channel(self.host))
        return self.stub.embed_query(Query(text = text, query_inst=self.query_instruction)).query
    

class Prompt_compressor():
    host: str
    stub: Any = None
        
    def __init__(self, host='localhost:50051'):
        self.host = host
        
    def compressor(self, files, instruction="", question=""):
        if files == "":
            return {
                'compressed_prompt': "", 
                'origin_tokens': 0, 
                'compressed_tokens': 0, 
                'ratio': 0
            }
        if self.stub == None:
            self.stub = compressed_promptStub(grpc.insecure_channel(self.host))
        result = self.stub.compress(Compress(docs=files, instruction=instruction, question=question))
        return {
            'compressed_prompt': result.docs, 
            'origin_tokens': result.origin_tokens, 
            'compressed_tokens': result.compressed_tokens, 
            'ratio': result.ratio
        }


_DEFAULT_SUMMARIZER_TEMPLATE = """
Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.
Keep the summary precise, retain the focus of the conversation as much as possible.
EXAMPLE
Current summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good.

New lines of conversation:
Human: Why do you think artificial intelligence is a force for good?
AI: Because artificial intelligence will help humans reach their full potential.

New summary:
The human asks what the AI thinks of artificial intelligence. The AI thinks artificial intelligence is a force for good because it will help humans reach their full potential.
END OF EXAMPLE

Current summary:
{summary}

New lines of conversation:
{new_lines}

New summary:"""
SUMMARY_PROMPT = PromptTemplate(
    input_variables=["summary", "new_lines"], template=_DEFAULT_SUMMARIZER_TEMPLATE
)

class SummarizerMixin(BaseModel):
    """Mixin for summarizer."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    llm: BaseLanguageModel
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    summary_message_cls: Type[BaseMessage] = SystemMessage

    def predict_new_summary(
        self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        return chain.predict(summary=existing_summary, new_lines=new_lines)
    
    
class ICSummaryMemory(BaseChatMemory, SummarizerMixin):

    max_token_limit: int = 2000
    memory_key: str = "history"
    url: str = "localhost:27017/"
    mongo_db: str = "TA"
    mongo_collection: str = "ic"
    uid: str = "ic"
    db: Any = None
    
    def add_connect(self):
        if self.uid == None or self.uid == "":
            raise Exception("uid is required")
        if self.db == None:
            mongo = pymongo.MongoClient("mongodb://" + self.url)
            mongodb = mongo[self.mongo_db]
            self.db = mongodb[self.mongo_collection]
    
    def read(self):
        self.add_connect()
        result = self.db.find_one({'role': 'summary', 'uid': self.uid})
        if result and 'content' in result:
            return result['content']
        return ""
                    
    def write(self, content):

        self.add_connect()
        if self.db.find_one({'role': 'conversation', 'uid': self.uid}):
            self.db.update_one(
                {'role': 'conversation', 'uid': self.uid}, 
                {'$push': {'history': {'$each': [content], '$position': 0}}, "$set": {"time": time.time()}}
            )
        else:
            self.db.insert_one(
                {'role': 'conversation', 'uid': self.uid, 'history': [content], "time": time.time()}
            )
    @property
    def buffer(self) -> List[BaseMessage]:
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        buffer = self.buffer
        content = self.read()
        if content != "":
            first_messages: List[BaseMessage] = [
                self.summary_message_cls(content=content)
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix
            )
        return {self.memory_key: final_buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        super().save_context(inputs, outputs)
        self.prune()

    def prune(self) -> None:
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            content = self.predict_new_summary(pruned_memory, self.read())
            self.write(content)
    
    def clear(self) -> None:
        super().clear()
        self.write("")

class MongoDBSummaryMemory(BaseChatMemory, SummarizerMixin):

    max_token_limit: int = 2000
    memory_key: str = "history"
    url: str = "localhost:27017/"
    mongo_db: str = "TA"
    mongo_collection: str = "conversation"
    uid: str = None
    db: Any = None
    
    def add_connect(self):
        if self.uid == None or self.uid == "":
            raise Exception("uid is required")
        if self.db == None:
            mongo = pymongo.MongoClient("mongodb://" + self.url)
            mongodb = mongo[self.mongo_db]
            self.db = mongodb[self.mongo_collection]
    
    def read(self):
        self.add_connect()
        result = self.db.find_one({'role': 'summary', 'uid': self.uid})
        if result and 'content' in result:
            return result['content']
        return ""
                    
    def write(self, content):
        self.add_connect()
        if self.db.find_one({'role': 'summary', 'uid': self.uid}):
            self.db.update_one({'role': 'summary', 'uid': self.uid}, { "$set": { "content": content, "time": time.time() } })
        else:
            self.db.insert_one({'role': 'summary', 'uid': self.uid, 'content': content, "time": time.time()})
        
    @property
    def buffer(self) -> List[BaseMessage]:
        return self.chat_memory.messages

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        buffer = self.buffer
        content = self.read()
        if content != "":
            first_messages: List[BaseMessage] = [
                self.summary_message_cls(content=content)
            ]
            buffer = first_messages + buffer
        if self.return_messages:
            final_buffer: Any = buffer
        else:
            final_buffer = get_buffer_string(
                buffer, human_prefix=self.human_prefix, ai_prefix=self.ai_prefix
            )
        return {self.memory_key: final_buffer}

    @root_validator()
    def validate_prompt_input_variables(cls, values: Dict) -> Dict:
        prompt_variables = values["prompt"].input_variables
        expected_keys = {"summary", "new_lines"}
        if expected_keys != set(prompt_variables):
            raise ValueError(
                "Got unexpected prompt input variables. The prompt expects "
                f"{prompt_variables}, but it should have {expected_keys}."
            )
        return values

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        super().save_context(inputs, outputs)
        self.prune()

    def prune(self) -> None:
        buffer = self.chat_memory.messages
        curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
        if curr_buffer_length > self.max_token_limit:
            pruned_memory = []
            while curr_buffer_length > self.max_token_limit:
                pruned_memory.append(buffer.pop(0))
                curr_buffer_length = self.llm.get_num_tokens_from_messages(buffer)
            content = self.predict_new_summary(pruned_memory, self.read())
            self.write(content)
    
    def clear(self) -> None:
        super().clear()
        self.write("")


class MongoDBBufferMemory(BaseChatMemory):
    """Buffer for storing conversation memory."""

    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "history"
    url: str = "localhost:27017/"
    mongo_db: str = "TA"
    mongo_collection: str = "conversation"
    uid: str = None
    db: Any = None
        
    def add_connect(self):
        if self.uid == None or self.uid == "":
            raise Exception("uid is required")
        if self.db == None:
            mongo = pymongo.MongoClient("mongodb://" + self.url)
            mongodb = mongo[self.mongo_db]
            self.db = mongodb[self.mongo_collection]
    
    def read(self):
        self.add_connect()
        result = self.db.find_one({'role': 'conversation', 'uid': self.uid})
        if result and 'history' in result:
            return result['history']
        return ""
                    
    def write(self, content):
        self.add_connect()
        if self.db.find_one({'role': 'conversation', 'uid': self.uid}):
            self.db.update_one({'role': 'conversation', 'uid': self.uid}, {'$push': {'history': {'$each': [content], '$position': 0}}, "$set": { "time": time.time() }})
        else:
            self.db.insert_one({'role': 'conversation', 'uid': self.uid, 'history': [content], "time": time.time()})
    
    def clear_history(self):
        self.add_connect()
        if self.db.find_one({'role': 'conversation', 'uid': self.uid}):
            self.db.update_one({'role': 'conversation', 'uid': self.uid}, { "$set": { "history": [], "time": time.time() } })

    @property
    def memory_variables(self) -> List[str]:
        return [self.memory_key]
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(f"One output key expected, got {outputs.keys()}")
            output_key = list(outputs.keys())[0]
        else:
            output_key = self.output_key
        human = f"{self.human_prefix}: " + inputs[prompt_input_key]
        ai = f"{self.ai_prefix}: " + outputs[output_key]
        self.write((human, ai))

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        history = ""
        for conv in self.read():
            history += conv[0] + "\n" + conv[1] + "\n"
        return {self.memory_key: history}
    
    def get_history(self, start_index=-1, finish_index=-1) -> Dict[str, Any]:
        return {self.memory_key: self.read()}
    
    def clear(self) -> None:
        self.clear_history()


class AsyncIteratorCallbackHandler(AsyncCallbackHandler):
    """Callback handler that returns an async iterator."""

    queue: asyncio.Queue[str]

    done: asyncio.Event

    @property
    def always_verbose(self) -> bool:
        return True

    def __init__(self) -> None:
        self.queue = asyncio.Queue()
        self.done = asyncio.Event()

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        self.done.clear()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        if token is not None and token != "":
            self.queue.put_nowait(token)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        self.done.set()

    async def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        self.done.set()

    async def aiter(self) -> AsyncIterator[str]:
        wait_time = 0
        while self.done.is_set() and wait_time <= 5:
            time.sleep(0.1)
            wait_time += 0.1
        while not self.queue.empty() or not self.done.is_set():
            while self.queue.empty():
                time.sleep(0.1)
                if self.done.is_set():
                    break
            if self.done.is_set() and self.queue.empty():
                break
            yield self.queue.get_nowait()
        yield '\0\0\0\0\0'
