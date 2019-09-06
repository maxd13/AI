#!/usr/bin/env python
import time
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin


#tentamos integrar o classificador com o scikit-learn. Nao deu la tao certo, mas valeu a ideia.
class CustomBayes(BaseEstimator, ClassifierMixin):

    def __init__(self, p):
        if p and (p > 1 or p <= 0):
            raise ValueError("Parameter p must have value between 0 exclusive and 1 inclusive.")
        self.p = p  

    @classmethod
    def posterior(cls, X, x, p, Y=None, y=None):
        """ Calcula a posterior da variavel categorica X tomar valor x
            condicionada na variavel categorica Y tomar valor y e nos dados,
            a depender do parametro p, que mede a probabilidade
            de haverem exatamente n amostragens de X nao observadas dado que existem
            pelo menos n amostragens de X nao observadas. Esta tambem
            e' a probabilidade de que todas as amostras possiveis de X tenham
            sido observadas.

            Se a variavel Y for nula, espera-se que o parametro y tambem seja, e
            se houver um valor de Y mas nao de y, uma excecao sera gerada.
            Neste caso e' computada a posterior incondicional de X.

            Assume-se para efeitos de calculo que as variaveis sao independentes
            a priori, e que a prior de X e' uniforme.
        """
        if Y is not None and y is None: 
            raise ValueError("If the variable Y is given as an argument, the value y needs to be defined")
        n = len(X)
        counts = len(X[X == x]) if Y is None else n*len(X[(X == x) & (Y == y)])/len(Y[Y == y])
        unique = np.unique(X)
        probX = 1/len(unique)
        return p * sum((1-p)**i * (counts + i*probX)/(n + i) 
                    for i in range(int(1/p) + 1)
                    )


    def bayes_theorem(self, sample, y):
        """ Aplica o teorema de Bayes para calcular a
            "probabilidade condicional" da variavel dependente Y
            tomar o valor y para a amostra dada condiconado aos
            dados dos atributos do problema. O parametro p
            e' passado para a funcao de calculo das posteriors.

            Na realidade o que e' calculado nao e' a probabilidade condicional,
            e o sim o estimador de Bayes. i.e. o produto da likelyhood function
            com a probabilidade de Y=y.
        """
        probY = self.posterior(self.Y_, y, self.p)
        accumulator = 1
        for i in range(len(sample)):
            accumulator *= self.posterior(self.X_[:, i], sample[i], self.p, self.Y_, y)
        return accumulator*probY

    def fit(self, X, Y):
        """ Treina um novo classificador Bayesiano no dataset.
            O modelo treinado e' um objeto python que representa um estimador estatistico,
            dotado de uma funcao de previsao que calcula para um unico array ou lista de
            valores das variaveis independentes representando uma amostra a ser classificada,
            qual a classe a que deve pertencer. Esta funcao retorna o objeto que a chamou.
        """
        if len(Y) != len(X):
            raise ValueError("number of observations of dependent and independent variables must match.")
        

        self.X_ = X
        self.Y_ = Y
        self.unique_ = np.unique(Y)
        

        return self
    
    def predict(self, samples):
        """ Esta funcao retorna a classe prevista para um conjunto de amostras.
            Ela tambem guarda no atributo confidence__ a "probabilidade" da
            amostra fazer parte da classe predita.

            Na verdade nao e' bem a probabilidade que e' guardada neste
            atributo, e sim o estimador Bayesiano da classe. i.e. se voce dividir o estimador pela
            probabilidade da ocorrencia dos dados voce vai obter a probabilidade condicional de Y
            tomar o valor y para a amostra.
        """
        if len(samples[0]) != len(self.X_[0]):
            raise ValueError("size of sample array must be equal to the number of independent variables.")
        

        self.confidence__ = []
        predictions = []
        for sample in samples:
            estimator = ((y, self.bayes_theorem(sample, y)) for y in self.unique_)
            result, confidence = max(estimator, key=lambda x: x[1])
            predictions.append(result)
            self.confidence__.append(confidence)

        return predictions


    def score(self, X, Y, sample_weight=None):
        result = (X == Y)*self.confidence__
        return np.sum(result)/len(Y) if sample_weight is None else np.dot(result, sample_weight)/len(Y)


names = ["class",
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa"
        ]

used = ["class",
        "handicapped-infants",
        "physician-fee-freeze",
        "religious-groups-in-schools",
        "aid-to-nicaraguan-contras",
        "adoption-of-the-budget-resolution",
        "immigration",
        "education-spending"
       ]

dataset = pandas.read_csv("house-votes-84.csv", names=names, header=None)
dataset[dataset == 'n'] = 0
dataset[dataset == '?'] = 1
dataset[dataset == 'y'] = 2
dataset[dataset == 'democrat'] = 0
dataset[dataset == 'republican'] = 1
dataset = dataset.astype('int32')
matrix = dataset.loc[:, used].values

Y = matrix[:,0]
X = matrix[:,1:]

#Os dados das variaveis indenpendentes (X) e dependente (Y)
#serao separados em um conjunto de teste e de treinamento,
#baseado em um paramentro variavel, e um modelo
#bayesiano sera treinado para cada separacao.
#valores em split sao tuplas do tipo
#(X_train, X_validation, Y_train, Y_validation)

split = [model_selection.train_test_split(X, Y, test_size=i) for i in np.linspace(0.1, 0.5, 5)]

#aqui o parametro padrao e' MUITO menor do que nos outros modelos.
default_bayes = [('DFNB', CustomBayes(p=1/len(Y)) )]

#note que o modelo CNB1.0 faz previsoes completamente empiricas.
bayes_models = default_bayes + [(f"CNB{i:.2}", CustomBayes(p=i)) for i in np.linspace(0.1, 1, 10)]

builtin_models = []
builtin_models.append(('LR', LogisticRegression()))
builtin_models.append(('KNN', KNeighborsClassifier()))
builtin_models.append(('CART', DecisionTreeClassifier()))
builtin_models.append(('NB', GaussianNB()))
builtin_models.append(('SVM', SVC()))

final_models = [bayes_models[5], builtin_models[3]]

bayesian_scores = []
accuracy_scores = []
train_times = []
predict_times = []
cpu_train_times = []
cpu_predict_times = []
names = []
print("Custom Bayes tests")
for name, model in final_models:
    score1 = []
    score2 = []
    times1 = []
    times2 = []
    cpu_times1 = []
    cpu_times2 = []
    for index, data in enumerate(split):
        X_train, X_validation, Y_train, Y_validation = data

        cpu_t = time.process_time()
        t = time.time()
        model.fit(X_train, Y_train)
        t = time.time() - t
        cpu_t = time.process_time() - cpu_t
        times1.append(t)
        cpu_times1.append(cpu_t)
        
        cpu_t = time.process_time()
        t = time.time()
        predictions = model.predict(X_validation)
        t = time.time() - t
        cpu_t = time.process_time() - cpu_t
        times2.append(t)
        cpu_times2.append(cpu_t)

        #score = model.score(Y_validation, predictions)
        acc_score = accuracy_score(Y_validation, predictions)
        #score1.append(score)
        score2.append(acc_score)
        #print(f"{name} on split {index}:")
        #print(f"    accuracy: {acc_score}")
        #print(f"    full score: {score}")
    
    #bayesian_scores.append(score1)
    accuracy_scores.append(score2)
    train_times.append(times1)
    predict_times.append(times2)
    cpu_train_times.append(cpu_times1)
    cpu_predict_times.append(cpu_times2)
    names.append(name)
    #mean1 = np.mean(score1)
    mean2 = np.mean(score2)
    #score1 = np.std(score1)
    score2 = np.std(score2)

    tmean1 = np.mean(times1)
    tmean2 = np.mean(times2)
    times1 = np.std(times1)
    times2 = np.std(times2)

    cpu_tmean1 = np.mean(cpu_times1)
    cpu_tmean2 = np.mean(cpu_times2)
    cpu_times1 = np.std(cpu_times1)
    cpu_times2 = np.std(cpu_times2)
    
    print(f"means for {name}:")
    print(f"    accuracy: {mean2} (+-{score2})")
    #print(f"    full score: {mean1} (+-{score1})")
    print(f"    training Wall time: {tmean1} (+-{times1})")
    print(f"    training CPU time: {cpu_tmean1} (+-{cpu_times1})")
    print(f"    prediction Wall time: {tmean2} (+-{times2})")
    print(f"    prediction CPU time: {cpu_tmean2} (+-{cpu_times2})")


# Comparar tempos
#fig, axs = plt.subplots(1, 2, constrained_layout=True, sharey=True)
#fig.suptitle('Tempos de Previsao')
##axs[0] = fig.add_subplot(111)
#axs[0].boxplot(predict_times)
#axs[0].set_xticklabels(names)
#axs[0].set_title("Tempo de Parede")
#axs[1].boxplot(cpu_predict_times)
#axs[1].set_title("Tempo de CPU")
#axs[1].set_xticklabels(names)
#plt.show()

## Comparar modelos por acuracia
fig = plt.figure()
fig.suptitle('Final Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(accuracy_scores)
ax.set_xticklabels(names)
plt.show()


#fig, axs = plt.subplots(2, 4, constrained_layout=True)
#axs[0,0].bar(["democrat", "republican"], dataset["class"].value_counts(), color=['blue', 'red'])
#axs[0,0].set_title("class")
#for i in range(1, 8):
#    x, y = 0 if i < 4 else 1, i % 4
#    data = dataset[used[i]].value_counts(sort=False)
#    aux = data[0]
#    data[0] = data[1]
#    data[1] = aux
#    axs[x, y].bar(['n', '?', 'y'], data, color=['red', 'yellow', 'green'])
#    axs[x, y].set_title(used[i])
#plt.suptitle("Frequencias dos atributos")
#plt.show()