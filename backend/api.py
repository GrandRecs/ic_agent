import time, asyncio, json, os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.exceptions import HTTPException
from sse_starlette.sse import EventSourceResponse
from fastapi.middleware.cors import CORSMiddleware

from IC import retrieval_service
from rpcllm import ICSummaryMemory, LLM
from student import student_service

MONGODB_URL = os.environ['MONGO_URI']
GPU_RUNTIME = os.environ['LLM_URI']
MILVUS_URL = os.environ['MILVUS_URI']

retrieval = retrieval_service(
    MILVUS_URL=MILVUS_URL,
    GPU_RUNTIME=GPU_RUNTIME
)

silent=LLM(host=GPU_RUNTIME, model_kwargs = {'temperature':0.6 ,'max_length': 256, 'top_k': 50}, uid="ic", stream_out=False)
ic_memory = ICSummaryMemory(url=MONGODB_URL, llm=silent, max_token_limit=200, input_key="input", output_key="output", memory_key="ic_history")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

user_data = {}

@app.get('/getChatHistory')
async def get_chat_history(request: Request):
    uid = request.headers.get('uid', None)
    if uid is None:
        return JSONResponse(status_code=400, content={"error": "User not found or UID not provided"})
    if uid not in user_data:
        user_data[uid] = student_service(uid, GPU_RUNTIME=GPU_RUNTIME, MONGODB_URL=MONGODB_URL, ic_memory=ic_memory, retrieval=retrieval)
    chat_history = user_data[uid].full.get_history()['full_history']
    return JSONResponse(status_code=200, content={"chat_history": chat_history})

@app.post('/postRequestToLLM')
async def post_request_to_llm(request: Request):
    data = await request.body()
    data = json.loads(data.decode("utf-8"))
    uid = request.headers.get('uid', None)
    if uid is None:
        return JSONResponse(status_code=400, content={"error": "User not found or UID not provided"})
    if "question" in data:
        question = data['question']
        if uid not in user_data:
            user_data[uid] = student_service(uid, GPU_RUNTIME=GPU_RUNTIME, MONGODB_URL=MONGODB_URL, ic_memory=ic_memory, retrieval=retrieval)
        tic = time.perf_counter()
        result = user_data[uid].query(question)
        toc = time.perf_counter()
        response = {
            "question": question,
            "result": result['output'],
            "execution_time": f"{toc-tic:0.1f}s"
        }
        return JSONResponse(status_code=200, content=response)
    else:
        return JSONResponse(status_code=400, content={"error": "parameter 'question' are required"})

@app.post('/getRequestFromLLM')
async def get_request_from_llm(request: Request) -> StreamingResponse:
    uid = request.headers.get('uid', None)
    data = await request.body()
    data = json.loads(data.decode("utf-8"))
    if uid is None or "question" not in data:
        raise HTTPException(status_code=400, detail={"error": "User not found or UID not provided"})
    if uid not in user_data:
        user_data[uid] = student_service(uid, GPU_RUNTIME=GPU_RUNTIME, MONGODB_URL=MONGODB_URL, ic_memory=ic_memory, retrieval=retrieval)
    question = data['question']
    async def event_stream():
        async for token in user_data[uid].stream(question):
            yield f"{token}"
            await asyncio.sleep(0)
    return StreamingResponse(event_stream(), media_type='text/event-stream')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=65500, workers=5)
    
