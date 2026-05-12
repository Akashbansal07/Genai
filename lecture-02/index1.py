from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key= os.getenv("OPEN_API_KEY"))

def ask_gpt(user_message, system ="You are a helpful assistant", temperature=1.0):
 
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=temperature,
        max_tokens = 1024,


        messages=[
            {
                "role": "system",
                "content": system
            },
            {
                "role":"user",
                "content":user_message
            }
        ]
    )
    return response.choices[0].message.content



if __name__ == "__main__":
    system_prompt="You are a tourist guide, answer concisely in points."

    result = ask_gpt(
        user_message="Tell me about hyderabad",
        system= system_prompt,
        temperature=0
    )

    print(result)


