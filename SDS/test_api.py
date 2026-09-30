import requests

url = "http://127.0.0.1:8000/predict/supervised"

data = {
    "Duration": 120.5,
    "Src_Pt": 443,
    "Dst_Pt": 52345,
    "Bytes": 4500,
    "bytes_per_packet": 150
}

response = requests.post(url, json=data)
print(response.json())
