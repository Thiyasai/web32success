import requests

URL = "https://web32success.onrender.com/health"

try:
    print(f"Checking: {URL}")
    response = requests.get(URL, timeout=10)
    print("Status Code:", response.status_code)
    print("Response:", response.text)

except Exception as e:
    print("Error:", e)

