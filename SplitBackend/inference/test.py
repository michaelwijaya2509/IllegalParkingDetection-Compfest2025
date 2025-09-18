import requests

url = "https://nominatim.openstreetmap.org/search"
params = {
    "q": "130 Quang Trung, Hải Châu, Đà Nẵng, Vietnam",
    "format": "json",
    "limit": 1
}
headers = {
    "User-Agent": "MyApp/1.0 (kenneth@example.com)"  # identify your app/email per Nominatim policy
}

req = requests.get(url, params=params, headers=headers, timeout=30)
result = req.json()

print(result)