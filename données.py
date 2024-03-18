import urllib.request
import tarfile
import os
import email.policy
import numpy as np

def download_and_extract_dataset(file_names, urls, download_directory, dataset_type):
    for (file_name, url) in zip(file_names, urls):
        file_path = os.path.join(download_directory, file_name)
        if not os.path.isfile(file_path):
            urllib.request.urlretrieve(url, file_path)
        tar_file = tarfile.open(file_path)
        
        # Remove the path by resetting it
        members = []
        for member in tar_file.getmembers():
            if member.isreg():
                member.name = os.path.basename(member.name) 
                members.append(member)
        tar_file.extractall(path=os.path.join(download_directory, dataset_type), members=members)
        tar_file.close()

def load_emails(directory, filename):
    
    with open(os.path.join(directory, filename), "rb") as f:
      
        return email.parser.BytesParser(policy=email.policy.default).parse(f)


root = "https://spamassassin.apache.org/old/publiccorpus/"

ham1_url = root + "20021010_easy_ham.tar.bz2"

ham3_url = root + "20030228_easy_ham_2.tar.bz2"

ham5_url = root + "20030228_hard_ham.tar.bz2"

ham_url = [ham1_url, ham3_url, ham5_url]

ham_filename = ["ham1.tar.bz2", "ham3.tar.bz2", "ham5.tar.bz2"]

spam1_url = root + "20021010_spam.tar.bz2"

spam4_url = root + "20050311_spam_2.tar.bz2"

spam_url = [spam1_url, spam4_url]

spam_filename = ["spam1.tar.bz2", "spam4.tar.bz2"]

path = "./data/"

if not os.path.isdir(path):
 os.makedirs(path)

download_and_extract_dataset(spam_filename, spam_url, path, "spam")

download_and_extract_dataset(ham_filename, ham_url, path, "ham")


ham_filenames = [name for name in sorted(os.listdir("./data/ham")) if name != 'cmds']
spam_filenames = [name for name in sorted(os.listdir("./data/spam")) if name != 'cmds']

ham_emails = [load_emails("./data/ham", filename=name) for name in ham_filenames]
spam_emails = [load_emails("./data/spam", filename=name) for name in spam_filenames]

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

ham= np.array(ham_emails,dtype=object)
spam= np.array(spam_emails,dtype=object)