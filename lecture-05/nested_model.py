from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
from datetime import date
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate



load_dotenv()

llm = ChatOpenAI(model="gpt-4o" , temperature=0)


class Salary(BaseModel):
    # nested model for salary information
    min_amount: Optional[float] = Field(description="Minimum salary, null if not specified")
    max_amount: Optional[float] = Field(description="Maximum salary, null if not specified")
    currency:   str             = Field(description="Currency code: INR, USD, etc.")
    period:     str             = Field(description="annual, monthly, or hourly")

class JobPosting(BaseModel):
    job_title:    str          = Field(description="The job title or role name")
    company:      str          = Field(description="Company name")
    location:     str          = Field(description="City and country or Remote")
    experience:   str          = Field(description="Required years of experience")
    skills:       List[str]    = Field(description="List of required technical skills")
    salary:       Salary       = Field(description="Salary information") 
    remote:       bool         = Field(description="True if remote work is allowed")

job_llm = llm.with_structured_output(JobPosting)

job_text="""
Senior Python Developer — Bangalore or Remote

We are looking for a Senior Python Developer with 5+ years of experience
to join our AI team at DataFlow Technologies.

Requirements:
- Strong Python skills (FastAPI, Django)
- Experience with machine learning (PyTorch, scikit-learn)
- Knowledge of AWS or GCP
- LangChain experience is a plus

Compensation: ₹18,00,000 – ₹25,00,000 per annum
Remote-friendly position. Apply by 30th April 2025.
"""

job = job_llm.invoke(f"Extract job posting details: \n\n{job_text}")


print(f"job: {job.job_title}")
print(f"Location{job.location}")