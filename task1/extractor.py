import os
from typing import List, Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables from a .env file
load_dotenv()

# 1. Define the Pydantic Schema for Structured Output
class LineItem(BaseModel):
    description: str = Field(description="Description of the item or service purchased.")
    quantity: int = Field(description="Quantity purchased.")
    unit_price: float = Field(description="Price per individual unit.")
    total_price: float = Field(description="Total price for this line item.")

class InvoiceSchema(BaseModel):
    invoice_number: str = Field(description="The unique invoice identifier or number.")
    vendor_name: str = Field(description="Name of the company or vendor issuing the invoice.")
    client_name: Optional[str] = Field(description="Name of the client or customer being billed.")
    invoice_date: str = Field(description="Date the invoice was issued.")
    line_items: List[LineItem] = Field(description="List of purchased items or services.")
    subtotal: float = Field(description="Subtotal before taxes and discounts.")
    tax_amount: float = Field(description="Total tax applied.")
    total_amount: float = Field(description="Final total amount due.")

def extract_invoice_data(unstructured_text: str) -> InvoiceSchema:
    """
    Sends unstructured text to Gemini and extracts data matching the InvoiceSchema Pydantic model.
    """
    # Initialize the client. 
    
    client = genai.Client()

    # Construct the parameterized prompt
    prompt = f"""
    You are an expert financial document parser. Extract structured information from the following unstructured invoice text accurately. 
    Ensure all fields adhere strictly to the requested schema.

    Invoice Text:
    {unstructured_text}
    """

    # Call the model with structured output configuration
    response = client.models.generate_content(
        model='gemini-3.8-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=InvoiceSchema,
            temperature=0.2,  # Low temperature for precise extraction
        ),
    )

    # Return the automatically parsed and validated Pydantic model instance
    return response.parsed

if __name__ == "__main__":
    # Sample messy/unstructured invoice text block
    sample_invoice = """
    INVOICE #INV-2026-8842
    Apex Cloud Solutions LLC
    To: Global Logistics Partners
    Date: November 3, 2026

    Details of Charges:
    1. Dedicated Server Hosting - 3 months @ $400.00/mo -> Total: $1,200.00
    2. Managed Database Backup Setup -> Qty: 1, Unit Price: $350.00, Total: $350.00

    Subtotal: $1,550.00
    Estimated Tax (8%): $124.00
    Total Balance Due: $1,674.00
    """

    print("--- Executing LLM Structured Extraction Pipeline ---")
    structured_data = extract_invoice_data(sample_invoice)
    
    # Output the result as JSON string
    print(structured_data.model_dump_json(indent=2))