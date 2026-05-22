from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


upload_response = client.files.create(
    file=open("traing_data.json1"),
    purpose="fine-tune"
)
file_id = upload_response.id
print(f"file uploaded: {file_id}")


fine_tune_job= client.fine_tuning.jobs.create(
    training_files=file_id,
    model="gpt-4o-mini"
)

job_id = fine_tune_job.id
print(f"Fine-tune job started:{job_id}")


#check the status

job_status= client.fine_tunning.jobs.retrieve(job_id)
print(f"status: {job_status.status}")


fine_tune_model = job_status.fine_tune_model
print(f"fine tune mode: {fine_tune_job}")


response = client.chat.completions.create(
    model=fine_tune_mclearodel,
    messages=[
        {"role":"system","prompt":"system prompt"},
        {"role":"user","prompt":"user prompt"},
    ]
)