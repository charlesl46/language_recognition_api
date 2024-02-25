import requests,json

print(requests.post("http://127.0.0.1:5000/detect-language",json=json.dumps({"sentence" : "Le chat est rouge"})).json())