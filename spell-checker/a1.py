#############################################################################
# Търсене и извличане на информация. Приложение на дълбоко машинно обучение
# Стоян Михов
# Зимен семестър 2022/2023
#############################################################################

# Домашно задание 1
###
# За да работи програмата трябва да се свали корпус от публицистични текстове за Югоизточна Европа,
# предоставен за некомерсиално ползване от Института за български език - БАН
###
# Корпусът може да бъде свален от:
# Отидете на http://dcl.bas.bg/BulNC-registration/#feeds/page/2
# И Изберете:
###
# Корпус с новини
# Корпус от публицистични текстове за Югоизточна Европа.
# 27.07.2012 Български
# 35337  7.9M
###
# Архивът трябва да се разархивира в директорията, в която е програмата.
###
# Преди да се стартира програмата е необходимо да се активира съответното обкръжение с командата:
# conda activate tii
###
# Ако все още нямате създадено обкръжение прочетете файла README.txt за инструкции

import model
import numpy as np


def editDistance(s1, s2):
	# функцията намира разстоянието на Левенщайн със сливане и разделяне между два низа
	# вход: низовете s1 и s2
	# изход: минималният брой на елементарните операции ( вмъкване, изтриване, субституция, сливане и разцепване на символи) необходими, за да се получи от единия низ другия

	#############################################################################
	# Начало на Вашия код. На мястото на pass се очакват 15-30 реда
	if s1 == s2:
		return 0

	rows  = len(s1)+1
	cols = len(s2)+1
	
	d = np.empty((rows , cols), dtype=np.int32)
	
	for i in range(rows): # s1 = ab, s2 = "" => editDistance = len(s1) 
		d[i][0] = i

	for j in range(cols):
		d[0][j] = j
	
	for i in range(1, rows):
		for j in range(1, cols):
			delta = 0 if s1[i - 1] == s2[j - 1] else 1
			substitution = d[i-1][j-1] + delta
			insertion = d[i-1][j] + 1
			deletion = d[i][j-1] + 1

			merge = d[i-2][j-1] + 1 if i > 1 else np.Inf # ор => ф
			split = d[i-1][j-2] + 1 if j > 1 else np.Inf # ф => ор
			
			d[i][j] = min(substitution, insertion, deletion, merge, split)
			
	
	return d[-1][-1]
	
	#### Край на Вашия код
	#############################################################################


def operationWeight(a,b):
	#### Функцията primitiveWeight връща теглото на дадена елементарна операция
	#### Тук сме реализирали функцията съвсем просто -- връщат се фиксирани тегла, които не зависят от конкретните символи. При наличие на статистика за честотата на грешките тези тегла следва да се заменят със съответни тегла получени след оценка на вероятността за съответната грешка, използвайки принципа за максимално правдоподобие
	#### Вход: Двата низа a,b, определящи операцията.
	#### Важно! При изтриване и вмъкване се предполага, че празният низ е представен с None
	#### изход: Теглото за операцията

	if a == None and len(b) == 1:
		return 3.0		  # insertion
	elif b == None and len(a) == 1:
		return 3.0		  # deletion
	elif a == b and len(a) == 1:
		return 0.0		  # identity
	elif len(a) == 1 and len(b) == 1:
		return 2.5		  # substitution
	elif len(a) == 2 and len(b)==1:
		return 3.5		  # merge
	elif len(a) == 1 and len(b)==2:
		return 3.5		  # split
	else:
		print("Wrong parameters ({},{}) of primitiveWeight call encountered!".format(a,b))

def editWeight(s1, s2):
	#### функцията editWeight намира теглото между два низа
	#### За намиране на елеметарните тегла следва да се извиква функцията operationWeight
	#### вход: низовете s1 и s2
	#### изход: минималното тегло за подравняване, за да се получи от единия низ другия
	
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 15-30 реда
	
	rows  = len(s1)+1 # i - s1 index
	cols = len(s2)+1 # j - s2 index
	cost = np.zeros((rows , cols)) 

		
	for i in range(1,rows): # first column - all operations are deletes (empty s2)
		cost[i][0] = cost[i-1][0] + operationWeight(s1[i-1], None)

	for j in range(1,cols): # first column - all operation are inserts (empty s1)
		cost[0][j] = cost[0][j-1] + operationWeight(None, s2[j-1])
	
	
	for i in range(1, rows):
		for j in range(1, cols):
			subtitution = cost[i-1][j-1] + operationWeight(s1[i - 1], s2[j - 1])
			deletion = cost[i-1][j] + operationWeight(s1[i - 1], None)
			insertion = cost[i][j-1] + operationWeight(None, s2[j - 1])

			merge = cost[i-2][j-1] + operationWeight(s1[i-2:i],s2[j-1]) if i > 1 else np.Inf # ор => ф
			split = cost[i-1][j-2] + operationWeight(s1[i-1],s2[j-2:j]) if j > 1 else np.Inf # ф => ор
			
			cost[i][j] = min(subtitution, insertion, deletion, merge, split)
	
	return cost[-1][-1]
	
	#### Край на Вашия код
	#############################################################################


def generateEdits(q):
	### помощната функция, generate_edits по зададена заявка генерира всички възможни редакции на разстояние едно от тази заявка.
	### Вход: заявка като низ q
	### Изход: Списък от низове на разстояние 1 по Левенщайн със сливания и разцепвания
	###
	### В тази функция вероятно ще трябва да използвате азбука, която е дефинирана в model.alphabet
	###
	#############################################################################
	#### Начало на Вашия код. На мястото на pass се очакват 10-15 реда
	length = len(q)
	# list of splits of the query at each letter
	splits = [(q[:half], q[half:]) for half in range(length + 1)]
    
    # deleting the last letter of fst
	delete = [fst[:-1] + snd for fst, snd in splits[1:]]

    #insert a letter between the last two elements of fst
	insert = [fst + c + snd for fst, snd in splits for c in model.alphabet]

    #substitute last letter of fst
	sub = [fst[:-1] + c + snd for fst, snd in splits[1:] for c in model.alphabet if c != fst[-1]]

	# merge last two letters of fst into one
	merge = [fst[:-2] + c + snd for fst, snd in splits[2:] for c in model.alphabet if c != fst[-2] and c != fst[-1]]

	# split last letter of fst into two letters
	combos = [(a,b) for a in model.alphabet for b in model.alphabet]
	split = [fst[:-1] + a + b + snd for fst,snd in splits[1:] for a,b in combos if a != fst[-1] and b != fst[-1]]

	return set(sub + delete + insert + merge + split)
	#### Край на Вашия код
	#############################################################################




def generateCandidates(query,dictionary):
	### Започва от заявката query и генерира всички низове НА РАЗСТОЯНИЕ <= 2, за да се получат кандидатите за корекция. Връщат се единствено кандидати, за които всички думи са в речника dictionary.
		
	### Вход:
	###	 Входен низ query
	###	 Речник: dictionary

	### Изход:
	###	 Списък от двойки (candidate, candidate_edit_log_probability), където candidate е низ на кандидат, а candidate_edit_log_probability е логаритъм от вероятността за редакция -- минус теглото.
	
	def allWordsInDictionary(q):
		### Помощна функция, която връща истина, ако всички думи в заявката са в речника
		return all(w in dictionary for w in q.split())

	L=[]
	if allWordsInDictionary(query):
		L.append((query,0))
	for query1 in generateEdits(query):
		if allWordsInDictionary(query1):
			L.append((query1,-editWeight(query1,query)))
	return L


def correctSpelling(r, model, mu = 1.0, alpha = 0.9):
	### Комбинира езиковия модел model с кандидатите за корекция генерирани от generate_candidates за създаване на най-вероятната желана заявка за дадената оригинална заявка query.			Генераторът на кандидати връща и вероятността за редактиране.
	###
	### Вход:
	###	заявка: r,
	###	езиков модел: model,
	###	тегло на езиковия модел: mu
	###	коефициент за интерполация на езиковият модел: alpha
	### Изход: най-вероятната заявка

	def getScore(q,logEditProb):
	
		### Използва езиковия модел и вероятността за редакцията "log_edit_prob" за изчисляване на оценка за кандидатска "q". 
		# Използва `mu` като степен на тежест за Pr[q].
	
		### вход:
		### q (str): Заявка за кандидат.
		### log_edit_prob (float): Логаритъм от вероятност за редакция на дадената заявка (т.е. log Pr[r|q], където r е оригиналната заявка).
		### изход:
		### log_p (float): Краен резултат за заявката, т.е. логаритъм от вероятността за кандидат заявката.
		#############################################################################
		#### Начало на Вашия код. На мястото на pass се очакват 1-3 реда

		#logEditProb = Pr(r|q)
		#logq = Pr(q)
		logq = model.sentenceLogProbability(q.split(), alpha)

		## Pr[q|r] = Pr[r|q]*Pr[q]^mu
		## logPr[q|r] = logPr[r|q]+mu*logPr[q]
		return logEditProb + mu*logq
		
		#### Край на Вашия код
		#############################################################################

	###
	###
	#############################################################################
	#### Начало на Вашия код за основното тяло на функцията correct_spelling. На мястото на pass се очакват 2-5 реда
	cands = generateCandidates(r, model.kgrams[tuple()]) # (q, Pr(r|q))
	best = max([(c,getScore(c,p)) for (c,p) in cands], key=lambda x: x[1])
	return best[0] if best else None
	
	#### Край на Вашия код
	#############################################################################