import pickle
import os
import glob
import re
import pickle
datapath = '20061020_20131126_bloomberg_news'
ticker_lst = []


def clean_sentence(sentence):
    # remove symbols in title, and return the title string
    sentence = sentence.lower().strip()
    sentence = re.sub(r'[^A-Za-z0-9\s]', '', sentence)
    return sentence


def find_title(txt):
    # find the title of each news, and return the title string
     bos = txt.find('--')
     end = txt.find('--',bos+2)
     title = txt[bos+3:end]
     title = title.replace("\n","")
     title = clean_sentence(title)
     return title

def find_ticker(ticker_lst, news):
    # find the ticker in the given news, and return a list of tickers 
    result = []
    bingo = re.findall(r"\([A-Z0-9]+\)", news)
    if bingo != []:
        for b in bingo:
            ticker = b[1:-1] # remove ( and ) in the string
            if ticker not in bad and ticker not in result:  # remove tickers in bad tickers set
                result.append(ticker) # remove ( and ) in the string, and append the ticker to result
    return result


cnt = 0 
final = []

# create bad tickers set
# some news include strings such as (1) (2) (3), but they are not tickers! 
bad = set()
for i in range(100):
    bad.add(str(i))


# iterate through the folder
for infile in glob.glob(os.path.join(datapath,'*')):
    for dinfile in glob.glob(os.path.join(infile,'*')):
            cnt += 1
            review_file = open(dinfile,'r').read()
            tickers = find_ticker(ticker_lst, review_file) # call find_tickers function
            if tickers != []: # if any ticker is found
                title = find_title(review_file)
                date = infile.replace('20061020_20131126_bloomberg_news/', '')
                final.append([tickers, date, title])
                #print(final)
            if cnt % 1000 == 0:
                print(str(cnt) + "news are done!")
   
# write file
with open('./results/tdt.pkl','wb' ) as f:
    pickle.dump(final, f)
print("Total news: ", cnt)
print("Financial news: ", len(final))
