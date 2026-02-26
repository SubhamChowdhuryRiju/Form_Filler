import asyncio
import os
from langchain_openai import ChatOpenAI
from browser_use import Agent

# For this to work, the user needs to set OPENAI_API_KEY
async def run():
    # Prompt user for API key if missing
    if not os.getenv("OPENAI_API_KEY"):
        api_key = input("Enter your OPENAI_API_KEY for browser-use: ")
        os.environ["OPENAI_API_KEY"] = api_key

    url = "https://docs.google.com/forms/d/e/1FAIpQLSe_j-PUgyyIapIUrwAKE3x57wiZPRmNvgDcImbZUrGOf_Y-xQ/viewform?usp=publish-editor"
    
    agent = Agent(
        task=f"Go to {url} and list out all the form fields present. For each field, specify the label, the type (e.g. text, radio, checkbox, select), and any options if it is a multiple choice question.",
        llm=ChatOpenAI(model="gpt-4o-mini", max_retries=2),
    )
    
    result = await agent.run()
    print("Agent Result:")
    print(result)

if __name__ == '__main__':
    asyncio.run(run())
