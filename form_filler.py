import asyncio
import os
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from browser_use import Agent, Controller
from typing import List, Optional

# Browser-use requires a 'provider' attribute on the LLM which ChatGoogleGenerativeAI lacks
class CustomChatGoogle(ChatGoogleGenerativeAI):
    provider: str = Field(default="google")

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
    
    scan_task = f"""
    Go to the URL {url}.
    Find all the input fields, questions, checkboxes, dropdowns, and text areas in this form.
    Ignore standard navigation links or generic page text. Focus only on things the user needs to fill out or click to submit a form.
    
    Return a comprehensive JSON list of all these fields exactly matching the requested Pydantic schema.
    For radio buttons, checkboxes, or dropdowns, make sure to extract all the available 'options' the user can choose from.
    """

    scan_agent = Agent(
        task=scan_task,
        llm=llm,
        generate_responses=True, # Allow structured output extraction
    )

    print("Agent is actively scanning the page (this may take 15-30 seconds depending on the form complexity)...")
    history = await scan_agent.run()
    
    # The last action result should contain our extracted data if asked nicely, or we can parse the final text.
    # To reliably get structured data out of browser-use, we should extract the final string and parse JSON.
    final_result_text = history.final_result()
    
    if not final_result_text:
        print("Failed to extract form data. The agent did not return a result.")
        return

    print("\\n--- Agent Finished Scanning ---")
    
    # Fallback to a secondary LLM call to strictly parse the text into our JSON schema 
    # since browser-use final_result is just a string summary.
    parse_task_prompt = f"""
    Extract the form fields from this raw text and output strictly valid JSON matching the following schema:
    [
      {{ "label": "string", "type": "string", "options": ["string", "string"], "required": boolean }}
    ]
    Raw text:
    {final_result_text}
    """
    
    # We use a direct LLM call to guarantee JSON format
    from langchain_core.messages import HumanMessage
    msg = HumanMessage(content=parse_task_prompt)
    structured_llm = llm.with_structured_output(FormSchema)
    
    try:
        schema_obj = structured_llm.invoke([msg])
        fields = schema_obj.fields
    except Exception as e:
        print(f"Failed to parse the fields: {e}")
        print(f"Raw output was: {final_result_text}")
        return

    if not fields:
        print("No visible form fields found on this page.")
        return
        
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
