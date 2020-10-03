from nltk import FreqDist, ConditionalFreqDist, bigrams, corpus
from nltk.book import text1
from math import log
import numpy

### The amount of unique tokens in the text divided by the total tokens of the text. ###
def lexical_diversity(srcText):
    return len(set(srcText)) / len(srcText)

### We remove punctuation because bigrams including punctuation don't seem useful. ###
def bigram(srcText):
    lowered = [word.lower() for word in srcText]
    filtered = [word for word in lowered if word.isalnum()]
    return list(bigrams(filtered))

### Filter out stop words, non-words, and return the 50 most common words in the FreqDist ###
### Task Four ###
def freq_dist_filtered_most_common(srcText, amt=None):
    filtered = freq_dist_filter(srcText)
    fd = FreqDist(filtered)
    return fd.most_common(amt)

### Helper for FreqDist filtering ###
def freq_dist_filter(srcText):
    swords = corpus.stopwords.words('english')
    lowered = [word.lower() for word in srcText]
    filtered = [word for word in lowered if word not in swords and word.isalnum()]
    return filtered

### Generate a Conditional FreqDist, where a title is the condition ###
### and the words within the title are the events. ###
def conditional_dist():
    cfdist = ConditionalFreqDist()
    fileids = corpus.gutenberg.fileids()
    for id in fileids:
        condition = id
        filteredText = freq_dist_filter(corpus.gutenberg.words(condition))
        for word in filteredText:
            if word not in cfdist[condition]:
                cfdist[condition][word] = 0
            cfdist[condition][word] += 1
    return cfdist

### Generate the frequency that some word occurs over a corpus of documents. ###
def doc_freq_dict(cfdist):
    ### Get initial number of occurances over the cfdist ###
    dfreq = {}
    for title in cfdist:
        fdist = cfdist[title]
        for word in fdist:
            if word not in dfreq:
                dfreq[word] = 0
            dfreq[word] += 1

    ### Iterate back over and divide by number of books in the cfdist. ###
    ### Would do this while iterating above but I'm afraid of rounding errors. ###
    for key in dfreq:
        dfreq[key] /= len(cfdist)
    return dfreq

### Generate the term frequency for some word, based off a passed count. ###
def term_freq(cfdist, document, word, count):
    return cfdist[document][word] / count

### Generate a TF-IDF score for each word of each text of Gutenberg corpus. ###
def tfidf(cfdist, dfreq, corpus_count):
    tfidfdict = {}
    for title in cfdist:
        tfidfdict[title] = {}
        fdist = cfdist[title]
        count = len(freq_dist_filter(corpus.gutenberg.words(title)))
        for word in fdist:
            tf = term_freq(cfdist, title, word, count)
            tfidfdict[title][word] = tf * log(corpus_count/dfreq[word])
    return tfidfdict

### Generate two vectors from each tfid based off the size of dfreq. ###
### Dot product these vectors, and divide this by the product of ###
### each vector's magnitude. ###
def cosine_similarity(tfid1, tfid2, dfreq):
    v1 = []
    v2 = []
    for word in dfreq:
        if word in tfid1:
            v1.append(tfid1[word])
        else:
            v1.append(0)
        if word in tfid2:
            v2.append(tfid2[word])
        else:
            v2.append(0)
    dp = numpy.dot(v1, v2)
    mags = numpy.linalg.norm(v1) * numpy.linalg.norm(v2)
    return dp/mags

### Compute the lexical diversity of an intro text. ###
def task_one():
    print("Computing lexical diversity")
    moby = text1
    return lexical_diversity(moby)

### Generate bigrams for the text used in task_one(). ###
def task_two():
    print("Generating bigrams.")
    moby = text1
    return bigram(moby)

### Generate FreqDist for the text in task_one() and task_two(). ###
### Return the 50 most common symbols of the FreqDist. ###
def task_three():
    print("Generating FreqDist and 50 most common symbols using an intro text.")
    moby = text1
    return freq_dist_filtered_most_common(moby, 50)

### Generate a FreqDist as before, using a Gutenberg text. ###
### Return the 50 most common symbols of the FreqDist. ###
def task_four():
    print("Generating FreqDist and 50 most common symbols using a Gutenberg text.")
    emma = corpus.gutenberg.words('austen-emma.txt')
    return freq_dist_filtered_most_common(emma, 50)

### Create a Conditional FreqDist where Gutenberg fileIDs are the condition ###
### and words are the event. Filtered as in tasks 3 and 4. ###
def task_five():
    return conditional_dist()

### Compute the document frequency for each word in a Conditional FreqDist. ###
def task_six(cfdist):
    return doc_freq_dict(cfdist)

### Generate a TF-IDF score for each word in a Conditional FreqDist. ###
### Stored as a dictionary of dictionaries: {title: [word, tf-idf score]}. ###
def task_seven(cfdist, dfreq, corpus_count):
    return tfidf(cfdist, dfreq, corpus_count)

### Generate Cosine Similarity for two given TF_IDF dicts. Create two vectors with size ###
### n = # of words in lexicon (we use the total number of words in doc_freq). Compute ###
### the magntude of each vector and the dot product between the two. ###
def task_eight(tfidf1, tfidf2, dfreq):
    return cosine_similarity(tfidf1, tfidf2, dfreq)

### A simple function to look at the cosine similarity of two given works. ###
### If these books seem similar (cosine similarity >= .5, this book is recommended. ###
def task_nine(cosine):
    return cosine >= .5

def main():
    """"
    print("Task one: ")
    print(task_one())
    print("Task two: ")
    bigrams = task_two()
    print("Task two complete! Omitting task two results (see main to view results).")
    #print(bigrams)
    print("Task three: ")
    print(task_three())
    print("Task four: ")
    print(task_four())
    """
    print("Task five: ")
    cfdist = task_five()
    print("Task five complete! Omitting task five results (see main to view results).")
    #print(cfdist)
    print("Task six: ")
    dfreq = task_six(cfdist)
    print("Task six complete! Omitting task six results (see main to view results).")
    #printf(dfreq)
    print("Task seven: ")
    corpus_count = len(corpus.gutenberg.fileids())
    tfidfdict = task_seven(cfdist, dfreq, corpus_count)
    print("Task seven complete! Omitting task seven results (see main to view results).")
    #printf(tfidfdict)
    print("Task eight, comparing two of Austen's texts (emma, persuasion): ")
    tfidf1 = tfidfdict['austen-emma.txt']
    tfidf2 = tfidfdict['austen-persuasion.txt']
    cosine = task_eight(tfidf1, tfidf2, dfreq)
    print("Cosine similarity is: ", cosine)
    print("Task eight complete!")
    print("Task nine: ")
    result = task_nine(cosine)
    if result:
        print("This book seems similar to the one you liked, you may enjoy it!")
    else:
        print ("This book doesn't seem very similar, you may not enjoy it :(.")
if __name__ == "__main__":
    main()
