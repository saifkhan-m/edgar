import re
m = re.search(r'.*(?<=-).*', 'spam-egg')
print(m)
print(m.group(0))
