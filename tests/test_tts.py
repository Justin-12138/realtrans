import requests

BASE_URL = "http://localhost:9099"


def test_tts():
    response = requests.post(
        f"{BASE_URL}/tts",
        data={"text": "Hello, world!"}
    )
    assert response.status_code == 200
