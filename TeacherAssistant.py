from langchain_google_genai import ChatGoogleGenerativeAI
import os

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
# import google.generativeai as genai
#
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("can you solve integration of lnx over 1 to 10 with detail steps? use strictly mat notation dont write in terms of <sup>")
# print(response.text)


llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model="gemini-1.5-pro",
    temperature=0.8,
    timeout=None,
    max_retries=2,
)
response = llm.invoke("you are math teacher for class 10th cbse and you are assigned to make question paper for class test, test consists of 10 questions and it is strictly on topics 'real number',"
                      "'quadratic equations' of mcq type, provide hint for teacher in separate line so they can solve"
                      "raise the standard and make it 5 question difficult to solve"
                      "")
print(response.content)