import numpy as np
import os
import re
from punctuation_dict import get_punctuation_dict
def getRhymeDict(text):
	lines = [line.split() for line in text.split('\n') if line.split()]
	sonnets = []
	sonnet = []
	for line in lines:
		if len(line) == 1:
			# Only store sonnets with 14 lines
			if len(sonnet) == 14:
				sonnets.append(sonnet)
			sonnet = []
			continue
		sonnet.append(line)

		
	# This rhyme dictionary is a list of sets, where all the elements in each set rhyme with each other
	rhyme_dict = []
	punc_dict = get_punctuation_dict()
	cap_words = ["i'll", 'i', 'o']
	
	def process_word(word):
		'''
		This function takes as its input a word and returns the processed word by 
		getting rid of unnecessary punctuations / capitalizations. 
		''' 
		# Exception "I'll" - confusion with ill should be manually taken care of
		if word == "I'll":
			return word
		
		# Convert to lowercase and remove punctuations not part of a word
		word = punc_dict[re.sub(r'[^\w]', '', word.lower())]

		# Keep certain words capitalized
		if word in cap_words:
			word = word.capitalize()
			
		return word
		
	def add_to_rhyme_dict(w1, w2):
		group_contain_w1 = None
		group_contain_w2 = None
		for group in rhyme_dict:
			if w1 in group:
				group_contain_w1=group
				rhyme_dict.remove(group)
				continue
			if w2 in group:
				group_contain_w2=group
				rhyme_dict.remove(group)
							
		if not (group_contain_w1 or group_contain_w2):
			rhyme_dict.append({w1, w2})
		elif not group_contain_w1:
			group_contain_w2.add(w1)
			rhyme_dict.append(group_contain_w2)
		elif not group_contain_w2:
			group_contain_w1.add(w2)
			rhyme_dict.append(group_contain_w1)
		else:
			group_contain_w2.update(group_contain_w1)
			rhyme_dict.append(group_contain_w1)

	for sonnet in sonnets:
		# Get all the rhyming pairs in the first 3 stanzas
		for i in [0, 1, 4, 5, 8, 9]:
			word1 = process_word(sonnet[i][-1])
			word2 = process_word(sonnet[i+2][-1])
			add_to_rhyme_dict(word1, word2)
		# Last two rows of a sonnet rhyme
		add_to_rhyme_dict(process_word(sonnet[12][-1]), process_word(sonnet[13][-1]))
	
	rhyme_dict = [list(x) for x in rhyme_dict]
	return rhyme_dict
		
		
if __name__ == '__main__': 
	text = open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read()
	data = getRhymeDict(text)
	print(data)
	np.savetxt('ryhme_dictionary.txt', data)		
	