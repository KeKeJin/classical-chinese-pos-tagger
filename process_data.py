with open('test.conllu') as file:
    data = file.read()

first = data.find('KR2b0041',0)
newData = data[first:]

with open('17-test.conllu', 'w') as file:
    file.write(newData)

with open('test.conllu', 'w') as file:
    file.write(data[:first])