#
# Machine Learning
#
# 1. treinamento
#   - amostragem de dados para treinamento
#       . t1, pro
#       . t2, anti
#       . t3, anti
#       . t4, pro
#       . t5, pro
#       . t6, anti...
# 2. execução
#   - submete dados para avaliação
#       . tx, ?
#       . ty, ?

from textblob.classifiers import NaiveBayesClassifier
# Para podermos usarmos apenas `NaiveBayesClassifier` 
# e nao `classifiers.NaiveBayesClassifier` caso fizessemos
# apenas `import textblob`

# Dados para treinamento do classificador
treinamento = [
    ('Fora Dilma','pro'),
    ('Impeachment sem crime e golpe','anti'),
    ('Fica Dilma','anti'),
    ('Fora Temer','anti'),
    ('Fora PT','pro'),
    ('Vem pra rua','anti'),
    ('Tchau querida','pro'),
    ('Lula na cadeia','pro'),
    ('Golpe e o preco','pro'),
    ('Nao vai ter golpe','anti')
]

# Criacao do classificador
classificador = NaiveBayesClassifier(treinamento)


# Alguns textos para teste
textos = [ 'Brasil contra o golpe. Amanha vamos pra rua',
            'Tchau Dilma, ja vai tarde',
            'Nao vai ter golpe, vai ter Impeachment'
        ]

# Vamos classificar os textos
for t in textos:
    classe = classificador.classify(t)
    print("'%s' -> %s" % (t,classe))

# Alguns textos para avaliacao da acuracia do classificador
teste = [
    ('Golpe e misogeno','anti'),
    ('Brasil contra o golpe. Amanha vamos pra rua','anti'),
    ('Tchau Dilma, ja vai tarde','pro'),
    ('Nao vai ter golpe, vai ter Impeachment','pro'),
    ('Tchau querida. Pra democracia ou pra Dilma?','anti')
]

acuracia = classificador.accuracy(teste)
print('Acuracia',acuracia)

# Proximos passos
# 
# 21/10
# Vocês tentam avançar na classificação da
# base da Tayrine.
#
# 28/10
# 1. Continua se aprofundando em classificação de textos <-- escolha da turma --
# 2. Modularidade na análise de redes
# 3. Outra coisa
