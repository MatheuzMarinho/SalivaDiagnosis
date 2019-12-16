import random
import math
import numpy as np
import time
from random_forest import random_forest

#Velocidade maxima
v_range=[-6,6]

class ParticulaFeatureSelect:
    def __init__(self,qtd_colunas,dataset):        
        self.dimensao = qtd_colunas
        self.dataset = dataset
        self.lista_solucao_posicao = []
        self.lista_velocidade = []
        self.lista_pbest = []
        self.inicializacao_aleatoria()
        self.valor_pbest = -1
        self.lista_pbest = []
        self.valor_atual = self.valor_pbest
    
    def inicializacao_aleatoria(self):
        for i in range (self.dimensao):
            self.lista_solucao_posicao.append(random.randint(0, 1))
            self.lista_velocidade.append(random.uniform(v_range[0], v_range[1]))

    def atualiza_posicao(self):
        for i in range(self.dimensao):
            v =  self.sigmoid_function(self.lista_velocidade[i])
            r_number = random.random()
            if(r_number < v):
                self.lista_solucao_posicao[i] = 1
            else:
                self.lista_solucao_posicao[i] = 0
            

    def sigmoid_function(self,v):
        s = math.exp(v)/(math.exp(v) +1)
        return s

    
    def atualizacao_velocidade_global(self, w, lista_gbest,INERCIA, c1, c2):
        for i in range(self.dimensao):
            e1 = random.random()
            e2 = random.random()
            velocidade_cognitiva = c1*e1* (self.lista_pbest[i] - self.lista_solucao_posicao[i])
            velocidade_social = c2*e2* (lista_gbest[i] - self.lista_solucao_posicao[i])
            if(INERCIA == 'CLERC'):
                v = round(w * (self.lista_velocidade[i] + velocidade_cognitiva + velocidade_social),2)
                if v>v_range[1]:
                    v = v_range[1]
                elif v<v_range[0]:
                    v = v_range[0]
                self.lista_velocidade[i] = v
            else:
                v= round(w * self.lista_velocidade[i] + velocidade_cognitiva + velocidade_social,2)
                if v>v_range[1]:
                    v = v_range[1]
                elif v<v_range[0]:
                    v = v_range[0]
                self.lista_velocidade[i] = v
            

        
    def avaliacao_solucao(self):
        self.valor_atual = random_forest.train_pso(self.lista_solucao_posicao,self.dataset)
        if(self.valor_atual > self.valor_pbest):
            self.valor_pbest= self.valor_atual
            self.lista_pbest= list(self.lista_solucao_posicao)

def decaimento_linear(iteracao_atual):
    w_max = 0.9
    w_min = 0.4
    return (w_max- w_min)*((quantidade_max_iteracao - iteracao_atual)/quantidade_max_iteracao)+w_min

def __get_fitness(individual_values, dataset):
    return random_forest.train(individual_values, dataset)

def execute(dataset, config):
    INERCIA = config.inertia_coeff
    valor_gbest= -1
    lista_gbest = []
    lista_fitness = []
    lista_best_features = []
    swarm = []

    #Quantidade de Colunas - remove a coluna de target
    qtdColunas = dataset.shape[1] - 1
    
    #TIME
    execution_time = 0

    #QTD de iterações sem melhorias
    unimproved_iterations_limit = 10
    unimproved_iterations = 0

    

    for i in range(config.population_size):
        swarm.append(ParticulaFeatureSelect(qtdColunas,dataset))


    for i in range(config.max_iterations):
        #TIME
        iter_start = time.time()
        #####
        print("ITERCAO: " ,i , " INERCIA: " + INERCIA)
        if INERCIA == 'CONSTANTE':
            w = 0.8
        elif INERCIA == 'LINEAR':
            w = decaimento_linear(i)
        else:
            w = 0.72984


        for j in range(config.population_size):
            swarm[j].avaliacao_solucao()

            if  swarm[j].valor_atual > valor_gbest:
                valor_gbest = swarm[j].valor_atual
                lista_gbest = list(swarm[j].lista_solucao_posicao)
                print('New best individual with fitness ' + str(valor_gbest))
                unimproved_iterations = 0
    

        for j in range(config.population_size):
            swarm[j].atualizacao_velocidade_global(w, lista_gbest, INERCIA, config.cognitive_coeff, config.social_coeff)              
            swarm[j].atualiza_posicao()
        
        
        lista_fitness.append(valor_gbest)
        lista_best_features.append(lista_gbest)
        if len(lista_fitness) > 2 and valor_gbest == lista_fitness[-2]:
            unimproved_iterations += 1
            print('No improvement in this iteration. ' +
                'Number of unimproved iterations: ' +
                str(unimproved_iterations) + '/' + str(unimproved_iterations_limit)
                )
        #TIME
        iter_time = time.time() - iter_start
        execution_time += iter_time
        print('Execution time: ' + str(execution_time) + ' s')
        print('*********************************\n')

        if unimproved_iterations == unimproved_iterations_limit:
            break

    selected_cols = [i for i, e in enumerate(lista_gbest) if e == 1]
    return selected_cols, lista_fitness, execution_time
