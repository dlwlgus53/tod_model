
import random

PUNCTUATIONS = ['.', ',', '!', '?', ';', ':']
def _insert_punctuation_marks(sentence, punc_ratio=0.3):
	words = sentence.split(' ')
	new_line = []
	q = random.randint(1, int(punc_ratio * len(words) + 1))
	qs = random.sample(range(0, len(words)), q)
	for j, word in enumerate(words):
		if j in qs:
			new_line.append(PUNCTUATIONS[random.randint(0, len(PUNCTUATIONS)-1)])
			new_line.append(word)
		else:
			new_line.append(word)
	new_line = ' '.join(new_line)
	return new_line

def run_aeda(user_text):
    return _insert_punctuation_marks(user_text)