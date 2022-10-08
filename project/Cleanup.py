import re

def cleanup(data):
    cleaned_data = []
    counter=0
    for line in data:
        # TODO Remove the sentiment of deleted rows
        counter = counter + 1
        try:
            line = re.sub("@", "", line) # remove @
            line = re.sub("#", "", line) # remove #
            # remove emojis: regex command from: https://gist.github.com/Alex-Just/e86110836f3f93fe7932290526529cd1
            line = re.sub(re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])'), "", line) # empojis
            line = re.sub(re.compile('<.*?>'), "", line) # remove html tags 
            line = re.sub('[^a-zA-Z\s]', '', line) # remove punctuations
            line = re.sub(r'\s+', ' ', line, flags=re.I) # remove unnecessary spaces
            # todo spellcheck()
            line = line.lower() # transform letters to lowercase
            cleaned_data.append(line)
        except:
            print("Deleting tweet nr: " + str(counter))
            cleaned_data.append("THIS TWEET WAS DELETED DURING CLEANUP")                   
    return cleaned_data