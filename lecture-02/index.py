from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key= os.getenv("OPEN_API_KEY"))

#response and request 
response = client.chat.completions.create(
    model="gpt-4o-mini",

    messages=[
        {
            "role":"user",
            "content":"How are you"
        }
    ]

)
print(response.choices[0].message.content)


usage= response.usage

print(f"Input tokens:  {usage.prompt_tokens}")
print(f"Output tokens:  {usage.completion_tokens}")
print(f"Total tokens:  {usage.total_tokens}")


input_cost =(usage.prompt_tokens/1_000_000)* 0.15
output_cost =(usage.completion_tokens/1_000_000)* 0.60


print(f"${input_cost:.6f} ${output_cost:.6f}")
print(f"${output_cost:.6f}")



