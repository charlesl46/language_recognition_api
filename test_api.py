import requests,json

if __name__ == "__main__":
    response = requests.post("http://127.0.0.1:5000/detect-language",json=json.dumps({"sentence" : "Le chat est rouge"}))
    print(response.json())