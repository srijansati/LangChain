from fastapi import FastAPI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama 
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

app = FastAPI(
    title= "Langchain Server",
    version = "1.0",
    description= "This is a simple API server"
)

#Ollama llama2
llm = Ollama(model = "llama2")

add_routes(
    app,
    ChatOpenAI(),
    path= "/openai"
)

model = ChatOpenAI()

prompt = ChatPromptTemplate.from_template("Write me a poem on {topic} of 20 words")
prompt1 = ChatPromptTemplate.from_template("Write me an essay on {topic} of 50 words")

add_routes(
    app,
    prompt|model,
    path= "/poem"
)

add_routes(
    app,
    prompt1|llm,
    path="/essay"
)

if __name__ == "__main__":
    uvicorn.run(app, host= "localhost", port= 8000)

