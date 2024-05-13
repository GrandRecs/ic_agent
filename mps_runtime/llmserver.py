import os, grpc, time, ast, torch, warnings

from threading import Thread
from concurrent import futures

from agent_pb2 import TextResponse, FloatListList, FloatList, Result
from agent_pb2_grpc import llmServicer, add_llmServicer_to_server, embedingServicer, add_embedingServicer_to_server, compressed_promptServicer, add_compressed_promptServicer_to_server

from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline, AutoConfig, TextIteratorStreamer

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.llms import HuggingFacePipeline
from prompt_compressor import PromptCompressor

warnings.filterwarnings('ignore')

port = 50051

print("Loading language model")
    
config = AutoConfig.from_pretrained("./models/neural-chat-7b-v3-3")
model = AutoModelForCausalLM.from_pretrained("./gguf/neural-chat-7B-v3-3-GGUF/neural-chat-7b-v3-3.Q4_K_M.gguf", hf=True, gpu_layers=1)
tokenizer = AutoTokenizer.from_pretrained("./models/neural-chat-7b-v3-3")
llm_lingua = PromptCompressor(model, tokenizer, config)

print("Loading embeding model")

embedding = HuggingFaceInstructEmbeddings(model_name="./models/bge-base-en-v1.5", model_kwargs={"device": "mps"})

session = {}

class llmServicer(llmServicer):
    model_kwargs = None
    pipeline_kwargs = None
    
    def stream(self, request, context):
        torch.mps.empty_cache()
        if request.id is None or request.id == "":
            return
        try:
            self.model_kwargs = ast.literal_eval(request.model_kwargs)
            self.pipeline_kwargs = ast.literal_eval(request.pipeline_kwargs)
            self.create(request.id)
            thread = Thread(target=session[request.id]['llm'], args=(request.prompt,))
            thread.start()
            for token in session[request.id]['streamer']:
    #             print(token, end="", flush=True)
                yield TextResponse(result=token)
            thread.join()
            torch.mps.empty_cache()
        except:
            for i in range(10):
                torch.mps.empty_cache()
        
    def create(self, uid):
        if uid not in session:
            session[uid] = {}
#         if 'pipe' not in session[uid] or 'llm' not in session[uid] or 'streamer' not in session[uid]:
        session[uid]['streamer'] = TextIteratorStreamer(tokenizer, Timeout=0.1, skip_prompt=True, skip_special_tokens=True)
        session[uid]['pipe'] = pipeline(
            "text-generation",
            model=model,
            tokenizer= tokenizer,
            streamer=session[uid]['streamer'],
            device_map="mps",
            max_new_tokens = 2048,
            pad_token_id=tokenizer.eos_token_id,
            batch_size=8,
        )
        session[uid]['pipe'].tokenizer.pad_token_id = model.config.eos_token_id
        session[uid]['llm'] = HuggingFacePipeline(pipeline=session[uid]['pipe'], model_kwargs=self.model_kwargs, pipeline_kwargs=self.pipeline_kwargs)
    

class embedingServicer(embedingServicer):
    def embed_documents(self, request, context):
        embedding.embed_instruction = request.embed_inst
        result = embedding.embed_documents(request.docs)
        return FloatListList(
            docs=[FloatList(query=float_data) for float_data in result]
        )
    
    def embed_query(self, request, context):
        embedding.query_instruction = request.query_inst
        result = embedding.embed_query(request.text)
        return FloatList(query=result)

class compressed_promptServicer(compressed_promptServicer):
    def compress(self, request, context):
        torch.mps.empty_cache()
        files = request.docs
        instruction = request.instruction
        question = request.question
        try:
            result = llm_lingua.compress_prompt(
                " ".join(files),
                instruction=instruction,
                question=question,
                target_token=500,
                condition_compare=True,
                condition_in_question='after',
                rank_method='longllmlingua',
                use_sentence_level_filter=False,
                context_budget="+100",
                dynamic_context_compression_ratio=0.4,
                reorder_context="sort"
            )
            torch.mps.empty_cache()
            return Result(docs=result['compressed_prompt'], origin_tokens=result['origin_tokens'], compressed_tokens=result['compressed_tokens'], ratio=result['ratio'])
        except:
            return Result(docs=" ".join(files), origin_tokens=0, compressed_tokens=0, ratio=0)
            for i in range(10):
                torch.mps.empty_cache()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_llmServicer_to_server(llmServicer(), server)
    add_embedingServicer_to_server(embedingServicer(), server)
    add_compressed_promptServicer_to_server(compressed_promptServicer(), server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    print("Server start on port {}".format(port))
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
