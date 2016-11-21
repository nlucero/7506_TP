import pyspark
import re
import math

MARGEN_COINCIDENCIA = 0.5
probabilidadClases = []

# Numero primo muy grande
p = 32452843

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

def hash_universal_string(string, a, m):
	h = ord(string[0])
	
	for i in range(1, len(string)):
		h = ((h * a) + ord(string[i])) % p
	
	return hash_int(h, a, m)

def hash_int(num, a, m):	
	return ((a * num) % p) % m
	
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


def process_train_row(x):
	# Obtenemos todas las palabras del texto
	textWords = re.sub("[^\w]", " ",  x[9]).split()
	
	# Filtramos las stop words
	nonStopWords = filter(lambda w: not(w in stopwords), textWords)
	nonStopWords = ' '.join(nonStopWords)
		
	# (Id, Text, Prediction)
	return x[0], nonStopWords, x[6]

def process_test_row(x):
	# Obtenemos todas las palabras del texto
	textWords = re.sub("[^\w]", " ",  x[8]).split()
	
	# Filtramos las stop words
	nonStopWords = filter(lambda w: not(w in stopwords), textWords)
	nonStopWords = ' '.join(nonStopWords)
		
	# (Id, Text)
	return x[0], nonStopWords
		

def trainingKNN(trainRDD, dimTHT, dimMH, shingleSize, hashesGroups, hashesPerGroup, hashFunction):
	return trainRDD.map(lambda x: (LSH(x[1], shingleSize, hashesGroups, hashesPerGroup, hashFunction, dimMH), x)) \
			.flatMap(lambda x: [(x[0][i],(x[1][0],tht(x[1][1],dimTHT,hashFunction),x[1][2])) for i in range(0, len(x[0]))])\
			.groupByKey()


def tht(review, dimTHT, hashTHT):
	output = [ 0 for i in range(0,k) ]

	# Utilizamos los valores a = 100 y b = 100 para la función de hash.
	for word in review:
		idx = hashTHT(word, 100, 100, k)
		output[idx] =+ 1


def LSH(review, shingleSize, hashesGroups, hashesPerGroup, hashFunction, dimMH):
	hashNumber = hashesGroups * hashesPerGroup
	kShingles = [ review[i:i+shingleSize] for i in range(0, len(review) - shingleSize + 1) ]
	minhashes = [ dimMH for i in range (0,hashNumber)]
	result = []

	for shingles in kShingles:
		for function in range(0, hashNumber):
			tmp = hashFunction(shingle, function, dimMH)
			if tmp < minhashes[hashNumber]:
				minhashes[hashNumber] = tmp

	# En este punto, la lista result tiene todos los minhashes.

	for grp in range(0,hashesGroups):
		result.append(hashFunction(minhashes[grp*cantHashesPorGrupo:grp*cantHashesPorGrupo+cantHashesPorGrupo-1],k))
		
	return result


# Obtiene el puntaje en base a los K mas cercanos. Puede ponderarase, ya que recibe los K registros mas cercanos completos.
def scoreKNN(list):
	aux = 0
	for i in range(0, len(list)-1):
		aux += list[i][4]
	return aux/len(list)

# Devuelve una lista con los K registros mas cercanos.	
def closestKNN(list, k):
	return None
	
def processKNN(knnRDD, test, k, shingleSize, hashFunction, cantGrupos, cantHashesPorGrupo):
	test.map(lambda x: (LSH(x[1], k, shingleSize, hashFunction, cantGrupos, cantHashesPorGrupo), x)) \
		.flatMap(lambda x: [(x[0][i],x[1]) for i in range(0, len(x[0]) - 1)])\
		.groupByKey()\
		.join(knnRDD)\
		.flatMap(lambda x: [(x[1][i],x[2]) for i in range(0, len(x[1]) - 1)])\
		.map(lambda x: (x[0][0],x[0][1],x[0][2],x[0][3],scoreKNN(closestKNN(x[2],k))))
	

def prediccionesCoinciden(prediccion1, prediccion2):
	return abs(prediccion1 - prediccion2) <= MARGEN_COINCIDENCIA

def addFrequency(scoringList, scoring):
	scoring = int(scoring)
	scoringList[scoring - 1] = scoringList[scoring - 1] + 1
	return scoringList

def normalize(vector):
	aux = 0
	for i in range(0, len(vector)):
		aux += vector[i]
	return [float(vector[i])/aux for i in range(0, len(vector))]

def trainingNB(trainRDD):
	return trainRDD.map(lambda x: (x[1].split(), x[2]))\
		.flatMap(lambda x: [(word, x[1]) for word in x[0]])\
		.map(lambda x: (x[0], addFrequency([0, 0, 0, 0, 0], x[1])))\
		.reduceByKey(lambda x,y: [x[i] + y[i] for i in range(0, len(x))])\
		.map(lambda x: (x[0], normalize(x[1])))
		
def processNB(trainRDD, test):
	return test.map(lambda x: (x[0], x[1].split()))\
		.flatMap(lambda x: [(word, x[0]) for word in x[1]])\
		.join(trainRDD)\
		.map(lambda x: x[1])\
		.reduceByKey(lambda x,y: [x[i] * y[i] for i in range(0, len(x))])\
		.map(lambda x: (x[0], [x[1][i] * probabilidadClases[i] for i in range(0, len(x[1]))]))\
		.map(lambda x: (x[0], getPrediction(x[1])))

def getPrediction(vectorFrecuencias):
	max = 0
	prediction = -1
	
	for i in range(0, len(vectorFrecuencias)):
		if (vectorFrecuencias[i] > max):
			max = vectorFrecuencias[i]
			prediction = i
	
	return (prediction + 1)
	
def main():
	# Loading the data.
	data = sc.textFile('data/train.csv')
	test = sc.textFile('data/test.csv')

	# Get the header.
	headerData = data.first()
	headerTest = test.first()

	# Extract the header from the dataset.
	data = data.filter(lambda line: line != headerData)
	test = test.filter(lambda line: line != headerTest)

	# Process each row
	data = data.map(lambda line: custom_split(line, ','))\
			   .map(lambda r: process_train_row(r))
	
	test = test.map(lambda line: custom_split(line, ','))\
			   .map(lambda r: process_test_row(r))
	
	# Armamos vector de probabilidades por clase	
	frecuenciasClases = data.map(lambda x: addFrequency([0, 0, 0, 0, 0], x[2]))\
		.reduce(lambda x,y: [x[i] + y[i] for i in range(0, len(x))])	
	global probabilidadClases
	probabilidadClases = normalize(frecuenciasClases)
	
	trainRDD = data
	faultRDD = None
	successExit = None
	
	for i in range(1, iterations):
		
		# Retroalimenta el trainRDD con los datos que se consiguen predecir correctamente.
		trainRDD = injectData(trainRDD, coincidenciasRDD)
		
		# "Entrenamiento" de KNN
		# Genera RDD con la forma (#Hash, (Id, Text, HelpfulnessNumerator, HelpfulnessDenominator, Prediction))
		knnRDD = trainingKNN(trainRDD, dimTHT, dimMH, shingleSize, hashesGroups, hashesPerGroup, hashFunction)
		
		# Procesamiento de KNN (obtenemos en el predictionesKNN los valores de las reviews)
		# (Id, Review, Prediction)
		predictionsKNN = processKNN(knnRDD, test, k, shingleSize, hashFunction, cantGrupos, cantHashesPorGrupo)
		
		# Entrenamiento de Naive-Bayes
		# Genera RDD con la forma (#Hash, (Text, (Freq. 0, Freq. 1, Freq. 2, Freq. 3, Freq. 4, Freq. 5)))
		naiveRDD = trainingNB(trainRDD)
		
		# Procesamiento de NB (obtenemos en el predictionSNB los valores de las reviews)
		# (Id, Review, Prediction)
		predictionsNB = processNB(naiveRDD, test)
		
		# Comparamos las 2 predicciones. Las que 'coinciden' las consideramos correctas y
		# ya parte de la solucion final. Las utilizamos para anexarlas al set de train. 
		# Las que no coinciden, las volvemos a calcular
		coincidenciasRDD = predictionsKNN.join(predictionsNB)\
										 .filter(lambda x: prediccionesCoinciden(x[1][0], x[1][1]))
		test = predictionsKNN.join(predictionsNB)\
								  .filter(lambda x: not prediccionesCoinciden(x[1][0], x[1][1]))
								  
		successExit = appendSuccessfuly(successExit, coincidenciasRDD)
	
	return successExit
	
if __name__ == "__main__":
    main()
