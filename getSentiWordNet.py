import collections
import sys
import csv
import pickle

def isFloat(num):
	try:
		float(num)
		return float(num)
	except:
		return -1

def readSentiWordNet(filePath):
	wordScore = dict()
	bl=0
	for line in open(filePath, 'r'):
		if line[0] != '#':
			fields = line.split("\t")
			#check if the line is formatted correctly	
			if len(fields) != 6:
				print "Parsing error, the file does not have 6 columns"
			#denotes if a word is a noun (n), adj (a), adverb (r) or a verb(v)  
			wordType = fields[0]
			posFlag = 0
			negFlag = 0
			posScore = isFloat(fields[2])
			negScore = isFloat(fields[3])
			if posScore == -1:
				posFlag = 1
			if negFlag == -1:
				negFlag = 1

			if posFlag == 0 and negFlag == 0:
				neuScore = 1 - posScore - negScore
				synSet = fields[4].split(" ")
				for words in synSet:
					wordAndRank = words.split(' ')
					if wordAndRank[0].find('#1') == len(wordAndRank[0])-2:
						wordKey = wordAndRank[0].strip('#1')+'#'+wordType
						print 'blah', wordKey
						if wordKey not in wordScore.keys():
							wordScore[wordKey] = dict()
							wordScore[wordKey]['type'] = wordType
							wordScore[wordKey]['posScore'] = posScore
							wordScore[wordKey]['negScore'] = negScore
							wordScore[wordKey]['neuScore'] = neuScore
					else:
						print "#2 #3 ..."
					
	return wordScore,bl
			
			
def serialize(wordFinalScore, objFile):
	pickle.dump( wordFinalScore, open( objFile, "wb" ) ) 

def main():
	filePath = "./SentiWordNet_3.0.0_20130122.txt"
	objFile = "SentiWordNet.p"
	wordScore,bl = readSentiWordNet (filePath)
	print bl
	serialize (wordScore, objFile)

if __name__ == '__main__':
	main()
	
	

	
