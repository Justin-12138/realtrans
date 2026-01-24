curl --location 'http://127.0.0.1:9088/v1/chat/completions' \
--header 'Authorization: Bearer sk-1234' \
--header 'Content-Type: application/json' \
--data '{
    "model":"HY-MT1.5-1.8B",
    "messages": 
        [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Translate the following segment into English, without additional explanation.\n\n欲买桂花同载酒，终不似少年游"
                }
            ]
        }],
    "stream": "true"
}'
