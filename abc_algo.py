import random
from roll import *
import numpy as np
import pandas as pd

#Functions and attributs to compute scores
def min_max_normalize(series):
    return (series - series.min()) / (series.max() - series.min())
    
values = np.array([6,12,1,8])
lengths = np.array([4,8,2,5])
a = [4,5,1,2]
b = [2,4,2,3]
c = [3,4,1,2] 
FEATURES = pd.DataFrame({
    'benefit' : values/lengths,
    'a' : a,
    'b' : b,
    'c' : c })
#normalized
FEATURES = FEATURES.apply(min_max_normalize)

def getScore(importance_rates, eps=0.001):
    """ return array of probs in depends of importance_rates input """
    f = FEATURES.copy() 
    #apply importance rate
    for i, col in enumerate(f.columns):
        f[col] = f[col]*importance_rates[i]
    scores = list(f.sum(axis=1)+eps)
    return scores


#In our case we want to maximize values, so we need to minimize fitness values, this is a potential candidate function:
def fit(eval):
  if eval >= 0:
    return 1/(1+eval)
  else:
    return 1 + abs(eval) 
      
class FoodSource(object):  
    def __init__(self, main_roll, benefit_rate):
        #init the roll variable
        self.source = main_roll.partition(0, main_roll.size-1)
        self.benefit_rate = benefit_rate
        #get scores array
        importance_rates = self.source.auto_constraint_rates(benefit_rate)
        scores = getScore(importance_rates)
        #place biscuits in depends of scores
        self.source.random_scoring_roll_rec_(scores)
        self.value = self.source.value
        self.fitness = fit(self.value)
        self.trial = 0
        
    def updateLocationRandom(self):
        """ used for scout phase, We compute a random benefit rate and we totaly reset current roll"""
        self.source.reset()
        self.benefit_rate = random.uniform(0,1)
        importance_rates = self.source.auto_constraint_rates(self.benefit_rate)
        scores = getScore(importance_rates)
        self.source.random_scoring_roll_rec_(scores)
        #self.source.randomized_roll_rec()
        self.value = self.source.value
        self.fitness = fit(self.value)
        self.trial = 0

    def updateLocation(self, partner):
        """ from a partner FoodSource input, we update the FoodSource if the new one gets a better fitness,
        """
        #select the new benefit rate between partner and current'rates
        new_benefit_rate = random.uniform(min(self.benefit_rate,partner.benefit_rate), max(self.benefit_rate,partner.benefit_rate))
        #select partition to change
        min_length = max(2, self.source.size//(self.trial+1))
        current_partition_dict = self.source.TournamentPartition(min_length)
        current_partition = current_partition_dict['partition']
        #calculate new score values
        importance_rates = current_partition.auto_constraint_rates(new_benefit_rate)
        new_scores = getScore(importance_rates)
        #generate a new partition
        new_partition = current_partition.partition(0, current_partition.size-1)
        new_partition.random_scoring_roll_rec_(new_scores)
        #greedy selection
        if new_partition.value > current_partition.value:
            #update
            self.source.updateFromPartition(new_partition, current_partition_dict['start'])
            self.value -= current_partition.value
            self.value += new_partition.value
            self.fitness = fit(self.value)
            self.trial = 0
        else:
            self.trial += 1



class ABC(object):
    def __init__(self, main_roll, colony_size, max_iter, limit):
        self.main_roll = main_roll.partition(0, main_roll.size-1)
        self.colony_size = colony_size
        self.max_iter = max_iter
        self.limit = limit
        self.sources = [FoodSource(main_roll, random.uniform(0, 1)) for i in range(colony_size)]
        self.best_solution = None
        self.best_answer = None

    def display(self):
        """ create a dataframe of our abc problem"""
        data = {} 
        data['foodsource'] = [str(i) for i in range(len(self.sources))]
        data['value'] =  [source.value for source in self.sources]
        data['fitness'] =  [source.fitness for source in self.sources]
        data['benefit rate'] = [source.benefit_rate for source in self.sources]
        data['trial'] = [source.trial for source in self.sources]
        display(pd.DataFrame(data))
        print('current best solution : ', self.best_solution)

    def displayAnswer(self):
        return
      
    def updateBest(self):
        """ catch the actual best solution and its answer, don't update if the previous is better"""
        total_values = np.array([source.value for source in self.sources])
        if not self.best_solution or total_values.max() > self.best_solution:
            #first solution
            self.best_solution = total_values.max()
            best_answer = self.sources[np.where(total_values == self.best_solution)[0][0]].source
            self.best_answer = best_answer.partition(0, best_answer.size-1, copy=True)
            

    def employed_phase(self):
        """create a new solution for each food location"""
        for i in range(self.colony_size):
            #select random partner, need to be different that the current food source explore 
            index = random.randint(0, len(self.sources)-1)
            while index == i:
                index = random.randint(0, len(self.sources)-1)
            
            self.sources[i].updateLocation(self.sources[index])

    def onlooker_phase(self):
        """create a new solution for each food location if the condition is valid,
          cond : r (random number between 0 and 1) < prob = fitness of source i / sum of all fitness"""
        total_fit = np.array([source.fitness for source in self.sources]).sum()
        prob = [source.fitness  / total_fit  for source in self.sources]
        i = 0
        n_update = 0
        #we have to update n times, n = size of the colony
        while n_update < self.colony_size:
            r = random.random()
            if r < prob[i]:
                #select random partner, need to be different that the current food source explore 
                index = random.randint(0, len(self.sources)-1)
                while index == i:
                  index = random.randint(0, len(self.sources)-1)
                self.sources[i].updateLocation(self.sources[index])
                #we count like update even if the source didn't change
                n_update += 1
                
            i = (i+1) % self.colony_size

    def scout_phase(self):
        """ forced to update food source with a trial > limit"""
        for i in range(self.colony_size):
            if self.sources[i].trial >= self.limit:    
                 self.sources[i].updateLocationRandom()

    def optimize(self, displayAll=False):
        """ run all the algo, if displayAll is set to True, 
        each iteration will be display
        """
        if displayAll:
            print('iteration {} :'.format(0))
            self.display()
            print('\n')
                
        for i in range(self.max_iter):
            self.employed_phase()
            self.onlooker_phase()
            self.updateBest()
            self.scout_phase()
              
            if displayAll:
                print('iteration {} :'.format(i+1))
                self.display()
                print('\n')
        
        return self.best_answer, self.best_solution
