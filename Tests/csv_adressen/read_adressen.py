content_string = open('adressen.csv', 'r').read()
content = [{content_string.split('\n')[0].split(';')[:-1][idx] : value for idx, value in enumerate(content_line)}
           for content_line in [line.split(';')[:-1] for line in content_string.split('\n')][1:]]
print(content)