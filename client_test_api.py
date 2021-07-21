import requests
res = requests.post('http://127.0.0.1:8000/do_ranking/',
                    files={'item':open('test_realdata/60.png', 'rb')},
                    params={'rank':5})
print(res.json())