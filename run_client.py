from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

chat_completion = client.chat.completions.create(
    model="ElicitResearch/llama-qa-ft-v0-fp8",
    # messages=[{"role": "user", "content": "What is the capital of New York?"}],
    # messages=[
    #     {"role": "user", "content": "Who is the 25th president of the United States?"}
    # ],
    messages=[
        {"role": "user", "content": "Write the 5 most populous cities in the world."},
    ],
)

print(chat_completion.choices[0].message.content)
