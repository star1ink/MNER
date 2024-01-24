import json

def racial2json():
    f = open("char2radical.txt","r",encoding="utf8")
    racial_dict = dict()
    while True:
        chars = f.readline().split(" ")
        if len(chars) != 2:
            break
        racial_dict[chars[0]] = chars[1][0]
    racial_json = json.dumps(racial_dict)
    f = open("racial.json","w",encoding="utf8")
    f.write(racial_json)
    return None

racial2json()






