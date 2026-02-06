from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:6001/v1",
    api_key="test"
)

def test_chat_completions():
    response = client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=[
            {"role": "user", "content": "你好，你是谁？"},
        ],
        max_tokens=512
    )
    assert response.choices[0].message.content is not None

test_chat_completions()