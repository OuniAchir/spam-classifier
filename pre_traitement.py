import re
from html import unescape
import collections
from nltk.stem import PorterStemmer
import string

# Cette fonction prétraite le corps d'un email
def preprocessing(email_contents):

  # Convert all letters to lowercase
  email_contents=email_contents.lower()
  # Remove HTML tags
  email_contents = re.sub('<[^<>]+>', ' ', email_contents)
  # Normalize URLs
  email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)
   # Normalize email addresses
  email_contents = re.sub('\S+@\S+', 'emailaddr', email_contents)
   # Normalize numbers
  email_contents = re.sub('\d+', 'nombre', email_contents)
   # Normalize dollar signs
  email_contents = re.sub('\$', 'dollar', email_contents)
   # Stem words
  stemmer = PorterStemmer()
  words = re.findall('\w+', email_contents)
  stemmed_words = [stemmer.stem(word) for word in words]
  email_contents = ' '.join(stemmed_words)
 # Remove non-words and punctuation, replace white spaces with a single space
   # Replace non-word characters with a space
  email_contents = re.sub(r'\W+', ' ', email_contents)
    #Remove punctuation
  email_contents = email_contents.translate(str.maketrans('', '', string.punctuation)) 
    # Replace newlines and tabs with a space
  email_contents= re.sub(r'\n|\t', ' ', email_contents)
    
    # Normalize whitespace
  email_contents = re.sub(r'\s+', ' ', email_contents).strip()
  return email_contents

#Convert html to text
def html_to_text(html):

    email_content= re.sub(r'<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    email_content = re.sub(r'<a\s.*?>', ' HYPERLINK ', email_content, flags=re.M | re.S | re.I)
    email_content = re.sub(r'<.*?>', '', email_content, flags=re.M | re.S)
    email_content = re.sub(r'(\s*\n)+', '\n', email_content, flags=re.M | re.S)
    
    return unescape(email_content) 

#Convert email to texte (lisibe)
def email_to_text(email):
    
    html = None
    for entity in email.walk():

        #Some emails have multiple parts, each part is handled separately
        entity_type = entity.get_content_type()
        if not entity_type in ("text/plain", "text/html"):
            continue
        
        try:
            content = entity.get_content()
            #Sometimes this is impossible for encoding reasons
        except: 
            content = str(entity.get_payload())
        if entity_type == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_text(html)
    
##la fonction qui fait le preprocessig de tous les emails
def preprocess(X): 
  emails_process =[]

  for email_content in X:
    email_content=email_to_text(email_content) or " "
    email_content=preprocessing(email_content)
    emails_process.append(email_content)

  return emails_process

#Creation list_vocabulaire
def list_vocabulaire2(X,k):
  list=X.split()
  vocabulaire=[]
  v = collections.Counter(list)
  keys = v.keys() # récupérer les clés de v
  for key in keys:
   if v[key]>k :
    vocabulaire.append(key) #ajouter les mots qui se repetent plus q k fois dans la lste vocabulaire
  return vocabulaire