import requests

url = "https://nominatim.openstreetmap.org/search"
params = {
    "q":  """
[out:json][timeout:25];
(
  node(around:{r},{lat},{lon})["amenity"~"hospital|clinic|police|fire_station|school|university|place_of_worship|bus_station"];
  node(around:{r},{lat},{lon})["public_transport"~"platform|stop_position|stop_area"];
  way(around:{r},{lat},{lon})["highway"];
  way(around:100,{lat},{lon})["highway"]["junction"="intersection"];
  way(around:{r},{lat},{lon})["access"="no"];
  way(around:{r},{lat},{lon})["busway"];
  way(around:100,{lat},{lon})["amenity"="parking"];
);
out body; >; out skel qt;
""",
    "format": "json",
    "limit": 1
}
headers = {
    "User-Agent": "MyApp/1.0 (kenneth@example.com)"  # identify your app/email per Nominatim policy
}

req = requests.get(url, params=params, headers=headers, timeout=30)
result = req.json()

print(result)
