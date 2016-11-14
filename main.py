import pyspark
import re

# Lista de stopwords
# Source:
stopwords = ["a", "about", "above", "above", "across", "after", 
"afterwards", "again", "against", "all", "almost", "alone", "along", 
"already", "also","although","always","am","among", "amongst", 
"amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone",
"anything","anyway", "anywhere", "are", "around", "as",  "at", "back",
"be","became", "because","become","becomes", "becoming", "been", 
"before", "beforehand", "behind", "being", "below", "beside", "besides",
"between", "beyond", "bill", "both", "bottom","but", "by", "call", 
"can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", 
"describe", "detail", "do", "done", "down", "due", "during", "each", 
"eg", "eight", "either", "eleven","else", "elsewhere", "empty", 
"enough", "etc", "even", "ever", "every", "everyone", "everything", 
"everywhere", "except", "few", "fifteen", "fify", "fill", "find", 
"fire", "first", "five", "for", "former", "formerly", "forty", 
"found", "four", "from", "front", "full", "further", "get", "give", 
"go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", 
"hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", 
"himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", 
"indeed", "interest", "into", "is", "it", "its", "itself", "keep", 
"last", "latter", "latterly", "least", "less", "ltd", "made", "many", 
"may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", 
"most", "mostly", "move", "much", "must", "my", "myself", "name", 
"namely", "neither", "never", "nevertheless", "next", "nine", "no", 
"nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", 
"of", "off", "often", "on", "once", "one", "only", "onto", "or", 
"other", "others", "otherwise", "our", "ours", "ourselves", "out", 
"over", "own","part", "per", "perhaps", "please", "put", "rather", "re",
"same", "see", "seem", "seemed", "seeming", "seems", "serious", 
"several", "she", "should", "show", "side", "since", "sincere", "six", 
"sixty", "so", "some", "somehow", "someone", "something", "sometime", 
"sometimes", "somewhere", "still", "such", "system", "take", "ten", 
"than", "that", "the", "their", "them", "themselves", "then", "thence", 
"there", "thereafter", "thereby", "therefore", "therein", "thereupon", 
"these", "they", "thickv", "thin", "third", "this", "those", "though", 
"three", "through", "throughout", "thru", "thus", "to", "together", 
"too", "top", "toward", "towards", "twelve", "twenty", "two", "un", 
"under", "until", "up", "upon", "us", "very", "via", "was", "we", 
"well", "were", "what", "whatever", "when", "whence", "whenever", 
"where", "whereafter", "whereas", "whereby", "wherein", "whereupon", 
"wherever", "whether", "which", "while", "whither", "who", "whoever", 
"whole", "whom", "whose", "why", "will", "with", "within", "without", 
"would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

try: 
	type(sc)
except NameError:
	sc = pyspark.SparkContext('local[*]')    

def custom_split(string, separator):	
	activated = True
	token_start_pos = 0
	return_value = []	
	
	for index in range(len(string) - 1):
		if string[index] == separator and activated:
			return_value.append(string[token_start_pos:index])
			token_start_pos = index + 1
		elif string[index] == '"':
			activated = not activated
	
	return_value.append(string[token_start_pos:])
	
	return return_value

def process_row(x):
	# Obtenemos todas las palabras del texto
	textWords = re.sub("[^\w]", " ",  x[9]).split()
	
	# Filtramos las stop words
	nonStopWords = filter(lambda w: not(w in stopwords), textWords)
	
	percentage = 0
	
	try:
		percentage = round(float(x[4])/float(x[5]), 2)
	except:
		# Si no se puede calcular el porcentaje, se usa 0
		percentage = 0
	
	# (Id, Text, HelpfulnessNumerator/HelpfulnessDenominator, Prediction)
	return x[0], x[9], nonStopWords, percentage, x[6]
		
		
def main():
	# Loading the data.
	data = sc.textFile('data/train.csv')

	# Get the header.
	header = data.first()

	# Extract the header from the dataset.
	data = data.filter(lambda line: line != header)

	# Get the marked values.
	data = data.map(lambda line: custom_split(line, ','))\
		.map(lambda r: process_row(r))\
	
	print data.take(5)
	
if __name__ == "__main__":
    main()
