import re

def extractsections(s):
    s = s.replace('\n', ' ').replace('I TEM','##Item')
    business = matchTheItem(s,'1','1A','BUSINESS','RIS')

    riskfactors = matchTheItem(s,'1A','1B','RIS','UNRES')

    legal = matchTheItem(s,'3','4','LEGAL','MINE')

    management = matchTheItem(s,'7','7A','MANAGE','QUANT')

    quantitative = matchTheItem(s,'7A','8','QUANT','FINAN')

    return business, riskfactors, legal, management, quantitative


def matchTiugdidudgdheItem(s):
    try:
        a = re.search("##Item\.*\s*1\.*\s*BUSINESS", s).start()
    except:
        a = re.search("##Item\s*\.*\s*"+itemNumstart+"\s*\.*\s*"+itemstart, s).start()
    b = re.search('##Item\.*\s*1A\.*\s*RIS', s).start()
    business = s[a:b]
    return business

def matchTheItem(s,itemNumstart,itemNumend,itemstart,itemend):

    try:
        a = re.search("##Item\s*\.*\s*"+itemNumstart+"\s*\.*\s*"+itemstart, s, re.IGNORECASE).start()
    except:
        a = re.search("##Item\s*\.*\s*"+itemNumstart+"\s*\.*\s*"+itemstart, s).start()
    try:
        b = re.search("##Item\s*\.*\s*"+itemNumend+"\s*\.*\s*"+itemend, s, re.IGNORECASE).start()
    except:
        if itemNumend=='4':
            try:
                b = re.search("##Item\s*\.*\s*" + '5, 6, 7' + "\s*\.*\s*", s).start()
            except:
                b = re.search("##Item\s*\.*\s*" + '5' + "\s*\.*\s*", s).start()
    item = s[a:b]
    return item


def clean_data(note):
    note = re.sub(pattern="((?i)<SEQUENCE>).*?(?=<)", repl='', string=note)
    note = re.sub(pattern="((?i)<FILENAME>).*?(?=<)", repl='', string=note)
    note = re.sub(pattern="((?i)<DESCRIPTION>).*?(?=<)", repl='', string=note)
    note = re.sub(pattern="(?s)(?i)<head>.*?</head>", repl='', string=note)
    #note = re.sub(pattern="(?s)(?i)<(table).*?(</table>)", repl='', string=note)
    note = re.sub(pattern="(?s)(?i)(?m)> +Item|>Item|^Item", repl=">##Item", string=note, count=0)
    note = re.sub(pattern="I TEM", repl=">##Item", string=note, count=0)
    note = re.sub(pattern="(?s)<.*?>", repl=" ", string=note, count=0)
    note = re.sub(pattern="&(.{2,6});", repl=" ", string=note, count=0)
    note = re.sub(pattern="(?s) +", repl=" ", string=note, count=0)
    return note


