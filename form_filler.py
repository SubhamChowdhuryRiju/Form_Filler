import asyncio
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ConfigDict
from browser_use import Agent, Controller
from typing import List, Optional

# Browser-use requires a 'provider' attribute on the LLM which ChatGoogleGenerativeAI lacks
# It also aggressively monkey-patches tracking methods onto the Pydantic object
class CustomChatGoogle(ChatGoogleGenerativeAI):
    model_config = ConfigDict(extra="allow")
    provider: str = Field(default="google")
    model_name: str = Field(default="gemini")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = getattr(self, "model", "gemini")

# Define the structured output we want the Agent to extract
class FormOption(BaseModel):
    label: str

class FormField(BaseModel):
    label: str
    type: str  # text, email, radio, checkbox, select, textarea, etc.
    options: Optional[List[str]] = None
    required: bool = False

class FormSchema(BaseModel):
    fields: List[FormField]


async def scan_and_fill(url):
    # Ensure API Key is set
    if not os.getenv("GEMINI_API_KEY"):
        api_key = input("Enter your GEMINI_API_KEY for browser-use: ")
        os.environ["GEMINI_API_KEY"] = api_key.strip()
        print("\\n")

    print(f"Starting browser-use agent to scan {url}...")
    
    # Initialize the LLM
    model_name = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    llm = CustomChatGoogle(model=model_name, temperature=0.0)

    # Note: We use gpt-4o because form understanding requires strong reasoning
    
    # Initialize Controller for extraction
    controller = Controller()

    # Parameter holding the parsed schema
    extracted_data: Optional[FormSchema] = None

    @controller.action("Save extracted form schema", param_model=FormSchema)
    def save_form_schema(schema: FormSchema):
        nonlocal extracted_data
        extracted_data = schema
        return "Successfully saved form schema! You should now stop."

    scan_task = f"""
    Go to the URL {url}.
    Find all the input fields, questions, checkboxes, dropdowns, and text areas in this form.
    Ignore standard navigation links or generic page text. Focus only on things the user needs to fill out or click to submit a form.
    
    Once you have identified all the fields, use the 'Save extracted form schema' action to save the data.
    For radio buttons, checkboxes, or dropdowns, make sure to extract all the available 'options' the user can choose from.
    """

    scan_agent = Agent(
        task=scan_task,
        llm=llm,
        controller=controller,
    )

    print("Agent is actively scanning the page (this may take 15-30 seconds depending on the form complexity)...")
    history = await scan_agent.run()
    
    if not extracted_data or not extracted_data.fields:
        print("Failed to extract form data or no fields found. The agent did not return a valid schema.")
        return

    fields = extracted_data.fields
        
    print(f"\\nFound {len(fields)} fields.\\n")
    
    values_to_fill = []
    
    # Ask user for values natively in Python terminal
    for i, field in enumerate(fields, 1):
        prompt = f"{i}. [{field.type.upper()}] {field.label}"
        if field.required:
            prompt += " *"
        
        if field.options and len(field.options) > 0:
            opts_str = ", ".join(field.options)
            prompt += f" (Options: {opts_str})"
        
        prompt += ": "
        val = input(prompt)
        values_to_fill.append({
            'label': field.label,
            'value': val,
            'type': field.type
        })
        
    print("\\nCommanding Agent to fill out the form automatically...")
    
    # Build the instruction for the execution agent
    fill_instructions = f"Navigate to {url} and fill out the form exactly according to these values:\\n"
    for item in values_to_fill:
        # Skip empty answers unless it's a checkbox we might want to ensure is unticked (though usually empty means skip)
        if not item['value'].strip():
            continue
        fill_instructions += f"- For the field labeled '{item['label']}', input or select: '{item['value']}'\\n"
    
    fill_instructions += "\\nAfter filling everything out, DO NOT SUBMIT. Just stop so the user can review it."

    fill_agent = Agent(
        task=fill_instructions,
        llm=llm
    )
    
    await fill_agent.run()
    
    print("\\nAgent has finished filling the form! Please review the browser window.")
    print("Press Enter in the terminal to close the script.")
    input()


if __name__ == "__main__":
    import sys
    url = input("Enter the URL of the form: ")
    if not url.startswith("http"):
        url = "https://" + url
    
    asyncio.run(scan_and_fill(url))
