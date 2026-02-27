import asyncio
import os
from langchain_openai import ChatOpenAI
from browser_use import Agent, Controller
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional

class CustomChatOpenAI(ChatOpenAI):
    model_config = ConfigDict(extra="allow")
    provider: str = Field(default="openai")
    model_name: str = Field(default="gpt-4o-mini")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = getattr(self, "model_name", "gpt-4o-mini")

class FormField(BaseModel):
    label: str
    type: str
    options: Optional[List[str]] = None
    required: bool = False

class FormSchema(BaseModel):
    fields: List[FormField]

# Initialize Controller
controller = Controller()

# Controller parameter holding action output
extracted_data: Optional[FormSchema] = None

@controller.action("Extract Form Schema", param_model=FormSchema)
def extract_form_schema(schema: FormSchema):
    global extracted_data
    extracted_data = schema
    return "Successfully extracted form schema! You should now stop."

# For this to work, the user needs to set OPENAI_API_KEY
async def run():
    # Prompt user for API key if missing
    if not os.getenv("OPENAI_API_KEY"):
        api_key = input("Enter your OPENAI_API_KEY for browser-use: ")
        os.environ["OPENAI_API_KEY"] = api_key

    url = "https://docs.google.com/forms/d/e/1FAIpQLSe_j-PUgyyIapIUrwAKE3x57wiZPRmNvgDcImbZUrGOf_Y-xQ/viewform?usp=publish-editor"
    
    agent = Agent(
        task=f"Go to {url} and find all the form fields present. For each field, specify the label, the type (e.g. text, radio, checkbox, select), and any options if it is a multiple choice question. Then use the 'Extract Form Schema' action to save the data.",
        llm=CustomChatOpenAI(model="gpt-4o-mini", max_retries=2),
        controller=controller,
    )
    
    await agent.run()
    
    print("Agent Result:")
    if extracted_data:
        print(f"Extracted {len(extracted_data.fields)} fields:")
        for f in extracted_data.fields:
            print(f"- {f.label} ({f.type})")
            if f.options:
                print(f"  Options: {f.options}")
    else:
        print("Failed to extract data.")

if __name__ == '__main__':
    asyncio.run(run())
