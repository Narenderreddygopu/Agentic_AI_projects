from langchain_openai import ChatOpenAI # helps to connect to the LLM - important role as a wrapper around the model
from langchain_core.prompts import ChatPromptTemplate # helps to create prompts for the LLM
from langchain_core.output_parsers import StrOutputParser # helps to parse the output from the LLM and get cleaner output

api_key = "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"

llm = ChatOpenAI(
    model = "openai/gpt-oss-120b:free",
    openai_api_key = api_key,
    openai_api_base = "https://openrouter.ai/api/v1",
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "Act as chef and give me the 5 best dishes from the {city}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser
city_by_user = input("Enter a city name: ")
result = chain.invoke({"city": city_by_user})
print(result)