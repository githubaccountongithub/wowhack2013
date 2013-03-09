import random

from map_reduce import MapReduce
from stemming.porter2 import stem

class doc(object):
	def __init__(self,rawText,docID):
		self.id = docID
		self.text = rawText

		#for entity in  [('&quot','"'),('&apos;',"'"),('&amp;','&'),('&lt;','<'),('&gt;','>')]:
		#	rawText = re.sub(entity[0],entity[1],rawText)
		#self.text = rawText.encode('ascii','ignore')	
	
		self.wordSet = set()
		self.centroidDist = 1.0        
		        
	def show(self):
		print self.text
		
	def addWord(self,index):
		self.wordSet.add(index)


class docCluster(object):

	def __init__(self, wordCount, dictSize):
		self.wordSet = set() # Holds the word indices of the cluster centroid.
		self.docIDs = []
		self.meanDistance = 1.0	
		while len(self.wordSet) < wordCount:
			word = random.choice(range(dictSize))
			self.wordSet.add(word)	
	
	# Jaccard distance: Similarity is the size of the intersection of a tweet and the centroid over the size of their union.
	# Subtract the similairity from one to get the distance. 
	def distanceToDoc(self,doc):
		return 1.0 - float(len(self.wordSet & doc.wordSet)) / (1+len(self.wordSet | doc.wordSet))

	# Add a dictionary of docs to the cluster, use IDs as keys.
	def addDocs(self,docDict):
		self.docIDs = docDict.keys() 
		
		# Fraction of the cluster's tweets that must contain a word for it to be added to the centroid.
		# If the centroid were the true mean of its tweets this would be 0.5, but the centroids would nearly always be empty!
		threshold = 0.1
		
		docCount = len(self.docIDs)
		minDocs = int(float(docCount)*threshold) 
		
		allWords = set() # Union of all the words in the cluster's docs.
		for doc in docDict.values():
			allWords |= doc.wordSet
		
		# Rebuild the centroid.
		self.wordSet = set()
		for word in allWords:
			if [ word in doc.wordSet for doc in docDict.values() ].count(True) > minDocs:
				self.wordSet.add(word)

		# Find the distance of each tweet in the cluster to the new centroid.
		for doc in docDict.values():
			doc.centroidDist = self.distanceToDoc(doc)

		# Mean distance of the tweets to the new centroid.
		self.meanDistance = sum([ doc.centroidDist for doc in docDict.values() ]) / docCount

# A set of docs, the clusters they belong to
class docCorpus(object):
  
	def __init__(self,items):
				
		self.docs = dict([ ( item[0],  doc(item[1],item[0]))  for item in items ])
		
		print self.docs
		
		self.freqTab = [] # List of (wordString,wordFrequency,unstemmed) tuples, most frequent words first.

		self.clusterSets = [] # Lists of clusters of increasing size.
		self.meanDistances = [1.0] # Mean centroid distance to each tweet for all cluster in a set.
		
	
	# Display all the docs.
	def show(self):
		for doc in self.docs.values():
			doc.show()
			print
	
	# Moderately decorous frequency table.
	def prettyTable(self):
		print 'Word frequencies:'
		# Jolly wheeze to get the terminal width. Wonder if it works on anything other than Linux...
		try:
			h, w = os.popen('stty size', 'r').read().split()
			w = int(w)
		except:
			w = 80
		colWidth = 9 + max([len(wrd[0]) for wrd in self.freqTab]) 
		cols = w / colWidth
		count = len(self.freqTab)
		rows = (count / cols)
		for row in range(rows):
			thisRow = ''
			for col in range(cols):
				index = col*rows + row
				if index < count:
					chunk = ' '.join([str(index),self.freqTab[index][3][0],'('+str(self.freqTab[index][1])+')'])
					thisRow += chunk + (colWidth-len(chunk))*' ' 
			print thisRow
	
	# Show the words in each cluster's centroid.
	def showClusters(self,index):
		clusters = sorted(self.clusterSets[index], key=lambda c: -len(c.docIDs))
		for i,cluster in  enumerate(clusters):
			print "Terms for cluster "+str(i)+", ("+str(len(cluster.docIDs))+" docs) mean centroid distance: "+str(cluster.meanDistance)
			words =  [ self.freqTab[word][3][0] for word in sorted(cluster.wordSet) ]
			print ' '.join(words) # Will these be vaguely relevant, or one of the aphorisms of Gertrude Stein? http://www.bartleby.com/140/
			print

class WordCount(MapReduce):
	def __init__(self,corpus):		
        	MapReduce.__init__(self)	
		self.corpus = corpus
		self.data = corpus.docs
	
	# Return a list of tuples: (tweetID,tweetText)
	def parse_fn(self, data):
		return [(key,data[key].text) for key in data.keys()] 	
	
	# Recieves a tweet's ID and it's text. Returns a list of (word,tweedID) tuples.
	def map_fn(self, key, val):
		words = val.split()
		return [ (stem(word), (1,key,word) ) for word in words ]
	
	# Receive all the IDs of tweets containing a given word. Return the word, its frequency and a list of unique tweet IDs. 
	def reduce_fn(self, word, values):
		count = 0
    		docs = []
    		unstemmed = []
    		for val in values:
        		count += val[0]
        		if val[1] not in docs:
				docs.append(val[1])
				if val[2] not in unstemmed:
					unstemmed.append(val[2])

        	return [(word, count, docs, unstemmed)]
	
	# Build the corpus' frequency table and give each tweet the indices of its words.
	def output_fn(self, output_list):
		for i,word in enumerate(sorted(output_list,key = lambda wrd: -wrd[1])):
			self.corpus.freqTab.append((word[0],word[1],word[2],word[3]))
			for doc in word[2]:
				self.corpus.docs[doc].addWord(i)


class kMeansIter(MapReduce):
	def __init__(self,corpus):
		MapReduce.__init__(self)	
		self.corpus = corpus
		self.data = corpus.docs
		self.totalDocs = len(self.data)
	
	# Return a list of (ID,tweet) tuples.
	def parse_fn(self, data):
		return [ (key,data[key]) for key in data.keys() ] 	
	
	# Given an ID and its tweet, return the index of the nearest cluster and the ID. 
	def map_fn(self, key, doc):
		minDistance = 1.0
		nearestCluster = 0
		for i,cluster in enumerate(self.corpus.clusterSets[-1]):
			distance = cluster.distanceToDoc(doc)
			if distance < minDistance:
				minDistance = distance
				nearestCluster = i
		
		tweet.centroidDist = minDistance
		
		return [(nearestCluster,key)]		
	
	# Given all the tweet IDs for a given cluster index, add thoses tweets to the cluster.
	def reduce_fn(self,clusterIndex,tweetIDs):
		tweetDict = dict([ (ID,self.corpus.docs[ID]) for ID in docIDs ]) # Send the tweets as a dictionary: {<tweetID>:<tweet>}
		thisCluster = self.corpus.clusterSets[-1][clusterIndex] # The given cluster in the latest set. 
		thisCluster.addDocs(tweetDict)

		return []

	def output_fn(self, output_list):
		
		meanDist = sum([ docs.centroidDist for doc in self.corpus.docs.values() ]) / self.totalDocs
		self.corpus.meanDistances[-1] = meanDist
		
		print "K-means iteration for "+str(len(self.corpus.clusterSets[-1]))+" centroids. Mean distance to centroids: "+str(meanDist)		

def doKmeans(corpus):
	
	# How similiar do successive mean cluster distances have to be before we declare convergence?
	nearlyOne = 0.999995
	
	totalWords = sum([len(doc.wordSet) for doc in corpus.docs.values()]) # Total word-count for the corpus.
	meanWords = totalWords / len(corpus.docs.values()) # Mean words per doc.
	
	corpusWords = len(corpus.freqTab) # Unique words in the corpus.
	
	bestCount = 2 # How many clusters yield the lowest mean centroid distance?
	clusterCount = 2
	getMoreClusters = True
	maxCount = max(len(corpus.docs) / 3, 3) # Don't let things degenerate to one tweet per cluster!
	
	# Try increasingly large cluster sets until convergence doesn't improve.
	while getMoreClusters and clusterCount <= maxCount:
	
		countMean = corpus.meanDistances[-1]
	
		# Start a new cluster set.
		corpus.clusterSets.append([])
		corpus.meanDistances.append(1.0)
	
		# Initialise the clusters to have similar numbers of words to the docs.
		# Would making the initial clusters less sparse do any good/harm?
		for i in range(clusterCount):	
			corpus.clusterSets[-1].append(tweetCluster(meanWords,corpusWords))

		moreIterations = True
	
		# Run k-means iterations for the current cluster set until convergence.	
		while moreIterations:
		#for i in range(10):
			lastMean = corpus.meanDistances[-1]
			k = kMeansIter(corpus)
			k.map_reduce()		
			if corpus.meanDistances[-1] > nearlyOne * lastMean:
				moreIterations = False
		
		# Do we have a new winner?
		if corpus.meanDistances[-1] < countMean:
			bestCount = clusterCount

		# Have we finished?
		if clusterCount > 2 and corpus.meanDistances[-1] > nearlyOne * countMean:
			getMoreClusters = False
		else:
			clusterCount += 1
			print	
	print
	print "Best results for "+str(bestCount)+" clusters."
	print
	corpus.showClusters(bestCount-2)
	
class kMeansIter(MapReduce):
	def __init__(self,corpus):
		MapReduce.__init__(self)	
		self.corpus = corpus
		self.data = corpus.docs
		self.totalDocs = len(self.data)
	
	# Return a list of (ID,tweet) tuples.
	def parse_fn(self, data):
		return [ (key,data[key]) for key in data.keys() ] 	
	
	# Given an ID and its tweet, return the index of the nearest cluster and the ID. 
	def map_fn(self, key, tweet):
		minDistance = 1.0
		nearestCluster = 0
		for i,cluster in enumerate(self.corpus.clusterSets[-1]):
			distance = cluster.distanceToDoc(tweet)
			if distance < minDistance:
				minDistance = distance
				nearestCluster = i
		
		tweet.centroidDist = minDistance
		
		return [(nearestCluster,key)]		
	
	# Given all the tweet IDs for a given cluster index, add thoses tweets to the cluster.
	def reduce_fn(self,clusterIndex,docIDs):
		docDict = dict([ (ID,self.corpus.docs[ID]) for ID in docIDs ]) # Send the tweets as a dictionary: {<tweetID>:<tweet>}
		thisCluster = self.corpus.clusterSets[-1][clusterIndex] # The given cluster in the latest set. 
		thisCluster.addDocs(docDict)

		return []

	def output_fn(self, output_list):
		
		meanDist = sum([ doc.centroidDist for doc in self.corpus.docs.values() ]) / self.totalDocs
		self.corpus.meanDistances[-1] = meanDist
		
		#print "K-means iteration for "+str(len(self.corpus.clusterSets[-1]))+" centroids. Mean distance to centroids: "+str(meanDist)		

def doKmeans(corpus,quiet=True):
	
	# How similiar do successive mean cluster distances have to be before we declare convergence?
	nearlyOne = 0.999995
	
	totalWords = sum([len(doc.wordSet) for doc in corpus.docs.values()]) # Total word-count for the corpus.
	meanWords = totalWords / len(corpus.docs.values()) # Mean words per tweet.
	
	corpusWords = len(corpus.freqTab) # Unique words in the corpus.
	
	bestCount = 2 # How many clusters yield the lowest mean centroid distance?
	clusterCount = 2
	getMoreClusters = True
	maxCount = max(len(corpus.docs) / 3, 3) # Don't let things degenerate to one tweet per cluster!
	
	# Try increasingly large cluster sets until convergence doesn't improve.
	while getMoreClusters and clusterCount <= maxCount:
	
		countMean = corpus.meanDistances[-1]
	
		# Start a new cluster set.
		corpus.clusterSets.append([])
		corpus.meanDistances.append(1.0)
	
		# Initialise the clusters to have similar numbers of words to the tweets.
		# Would making the initial clusters less sparse do any good/harm?
		for i in range(clusterCount):	
			corpus.clusterSets[-1].append(docCluster(meanWords,corpusWords))

		moreIterations = True
	
		# Run k-means iterations for the current cluster set until convergence.	
		while moreIterations:
		#for i in range(10):
			lastMean = corpus.meanDistances[-1]
			k = kMeansIter(corpus)
			k.map_reduce()		
			if corpus.meanDistances[-1] > nearlyOne * lastMean:
				moreIterations = False
		
		# Do we have a new winner?
		if corpus.meanDistances[-1] < countMean:
			bestCount = clusterCount

		# Have we finished?
		if clusterCount > 2 and corpus.meanDistances[-1] > nearlyOne * countMean:
			getMoreClusters = False
		else:
			clusterCount += 1
			if not quiet:
				print
	if not quiet:
		print
		print "Best results for "+str(bestCount)+" clusters."
		print
		corpus.showClusters(bestCount-2)

f = open('egtext')
txt = f.read()
f.close()
bits = txt.split('.')
docs = [bit for bit in enumerate(bits)]

corp = docCorpus(docs)
counter = WordCount(corp)
counter.map_reduce()
corp.prettyTable()
doKmeans(corp,False)	