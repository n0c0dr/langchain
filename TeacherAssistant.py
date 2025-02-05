from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
# from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
load_dotenv()

# model = ChatAnthropic(model='claude-3-opus-20240229')

## setting up llm
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#  LLm that i am using we can switch to other if required
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.8,
    timeout=None,
    max_retries=2,
)

## embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector = embeddings.embed_query("hello, world!")

## prompts
message=[
 SystemMessage("You are an teacher assistant for class 10 and your job is to prepare multiple choice question, answer and hint on given topic"),
 HumanMessage("Prepare 10 question on topic Quadratic Equation, make 5 of medium level and 5 of hard level, write question in mathematical term/notation and in last give answer and hint as well don;t use unnecessary notation")
]

response = llm.invoke(message)
print(response.content)

