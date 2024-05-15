import os, grpc, time, ast, torch, warnings, sys

from threading import Thread
from concurrent import futures

from agent_pb2 import TextResponse, FloatListList, FloatList, Result
from agent_pb2_grpc import llmServicer, add_llmServicer_to_server, embedingServicer, add_embedingServicer_to_server, compressed_promptServicer, add_compressed_promptServicer_to_server

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextStreamer, AutoConfig, TextIteratorStreamer

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from prompt_compressor import PromptCompressor

warnings.filterwarnings('ignore')

model_name = os.environ.get('MODEL') or "neural-chat-7b-v3-3"
embedding_name = os.environ.get('EMBEDING') or "bge-base-en-v1.5"

if not os.path.exists("../models/" + model_name):
    print(model_name + " does not exist in the ./models/")
    sys.exit(1)
if not os.path.exists("../models/" + embedding_name):
    print(embedding_name + " does not exist in the ./models/")
    sys.exit(1)

port = 50051

print("Loading language model")
    
config = AutoConfig.from_pretrained("../models/" + model_name)
tokenizer = AutoTokenizer.from_pretrained("../models/" + model_name)
model = AutoModelForCausalLM.from_pretrained(
    "../models/" + model_name,
    load_in_4bit=True, 
    bnb_4bit_use_double_quant=True, 
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16, 
    attn_implementation="flash_attention_2",
)
llm_lingua = PromptCompressor(model, tokenizer, config)

print("Loading embeding model")

embedding = HuggingFaceInstructEmbeddings(model_name="./models/" + embedding_name, model_kwargs={"device": "cuda"})

session = {}

class llmServicer(llmServicer):
    model_kwargs = None
    pipeline_kwargs = None
    
    def stream(self, request, context):
        torch.cuda.empty_cache()
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
            torch.cuda.empty_cache()
        except:
            for i in range(10):
                torch.cuda.empty_cache()
        
    def create(self, uid):
        if uid not in session:
            session[uid] = {}
#         if 'pipe' not in session[uid] or 'llm' not in session[uid] or 'streamer' not in session[uid]:
        session[uid]['streamer'] = TextIteratorStreamer(tokenizer, Timeout=0.1, skip_prompt=True, skip_special_tokens=True)
        session[uid]['pipe'] = pipeline(
            "text-generation", 
            device_map='cuda', 
            model=model, 
            tokenizer=tokenizer,
            streamer=session[uid]['streamer'],
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
        torch.cuda.empty_cache()
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
            torch.cuda.empty_cache()
            return Result(docs=result['compressed_prompt'], origin_tokens=result['origin_tokens'], compressed_tokens=result['compressed_tokens'], ratio=result['ratio'])
        except:
            for i in range(10):
                torch.cuda.empty_cache()

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
