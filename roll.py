import numpy as np
import pandas as pd
import random

biscuit0 =  {'symbol': '0', 'value': 6, 'length': 4, 'a': 4, 'b': 2, 'c': 3}
biscuit1 =  {'symbol': '1', 'value': 12, 'length': 8, 'a': 5, 'b': 4, 'c': 4}
biscuit2 =  {'symbol': '2', 'value': 1, 'length': 2, 'a': 1, 'b': 2, 'c': 1}
biscuit3 =  {'symbol': '3', 'value': 8, 'length': 5, 'a': 2, 'b': 3, 'c': 2}   
BISCUITS = [biscuit0, biscuit1, biscuit2, biscuit3]
# the smaller length of biscuits
MINLENGTH = 2

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
    
class Roll(object):
    #problem definition
    def __init__(self,size):
        self.size = size
        self.defects =  [{'a':0,'b':0,'c':0} for i in range(size)]
        self.biscuits = ['_' for i in range(size)]
        self.value = 0
        
    def getNumbersDefects(self, start=0, end=None):
        a, b, c = 0, 0, 0
        if not end:
            end = self.size
        for i in range(start, end):
            a += self.defects[i]['a']
            b += self.defects[i]['b']
            c += self.defects[i]['c']
        return {'a' : a, 'b' : b, 'c' : c}
        
    def partition(self,start,end, copy=False):
        """ return a partition of the Roll from pos start to end """
        roll_part = Roll(1+end-start)
        roll_part.defects =  self.defects[start:end+1].copy()
        if copy:
             roll_part.biscuits = self.biscuits[start:end+1].copy()
             roll_part.updateValue()
        return roll_part

    def addDefect(self, defect, pos):
        """ add the defect at the position pos in the Roll """
        self.defects[pos][defect] += 1
 

    def addBiscuit(self, biscuit, start):
        """ add the biscuit if it possible, return True, else do nothing and return False
        """
        end = start+biscuit['length']
        #over
        if end <= self.size and self.biscuits[start:end] ==  ['_' for i in range(start, end)]:
            defects = self.getNumbersDefects(start, end)
            #check constraint
            if biscuit['a'] < defects['a']:
                return False
            if biscuit['b'] < defects['b']:
                return False
            if biscuit['c'] < defects['c']:
                return False
            
            self.biscuits[start:end] =  [biscuit['symbol'] for i in range(start,end)]
            self.biscuits[start] = '[' + biscuit['symbol']
            self.biscuits[end-1] = biscuit['symbol'] + ']'
            self.value += biscuit['value']
            return True
            
        return False

    def deleteBiscuit(self, biscuit, start):
        i = start
        if self.biscuits[i] == '[' + biscuit['symbol']:
            while i <= start + biscuit['length'] - 1 and self.biscuits[i] != biscuit['symbol'] + ']':
                self.biscuits[i] = '_'
                i+=1
            
            self.value -= biscuit['value']
        
    def display(self, part=20):
        if len(self.defects) < 20:
            part = len(self.defects)
        info =  { str(i) : [self.defects[i],self.biscuits[i]] for i in range(self.size)}
        df =  pd.DataFrame(info)
        print('Total Value : ', self.value)
        for i in range(0, self.size // part):
            display(df[df.columns[i*part : i*part+part]])
        if self.size % part != 0:
            display(df[df.columns[(i+1)*part : (i+1)*part+(self.size % part)]])
         
    
    #manage random placing of bsicuits
    def randomized_roll_rec(self, start=0):
        """ place biscuit randomly in the roll """
        #remain space is to small to place any existing biscuit
        if len(self.biscuits) - start < MINLENGTH:
                return 
        biscuit =  random.sample(BISCUITS, 1)[0]
        already_tried = []
        
        while not self.addBiscuit(biscuit, start):
            #update current possibilities
            already_tried.append(biscuit)
            #no possibilities at this position, start from second pos from current start
            if len(already_tried) == 4:
                return self.randomized_roll_rec(start + 1)
        
            biscuit =  random.sample(BISCUITS, 1)[0]
               
        self.randomized_roll_rec(biscuit['length']+start)
        

    def random_scoring_roll_rec_(self, scores, start=0, max_iter=100):
        """ place biscuit randomly in the roll but in depends of probabilities """
        #remain space is to small to place any existing biscuit
        if len(self.biscuits) - start < MINLENGTH:
                return 
        #compute probabilities
        scores_ = scores.copy()
        probabilities = [(score) / sum(scores_) for score in scores_] 
    
        n_possibilities = len(BISCUITS)
        possible_biscuits = BISCUITS.copy()
        #pick a biscuit randomly and consider probabilities
        index = np.random.choice(n_possibilities, size=1, p=probabilities)[0]
        biscuit = possible_biscuits[index].copy()
        already_tried = []
        
        while not self.addBiscuit(biscuit, start):
            #update current possibilities
            already_tried.append(biscuit)
            #no possibilities at this position, start from second pos from current start
            if len(already_tried) == 4:
                return self.random_scoring_roll_rec_(scores,start+1)
                
            #update current possibilities
            n_possibilities -= 1
            possible_biscuits.pop(index)
                
            #recomputes probabilities
            scores_.pop(index)
            probabilities = [score / sum(scores_) for score in scores_]

            #choose an another posibilitie
            index = np.random.choice(n_possibilities, size=1, p=probabilities)[0]
            biscuit = possible_biscuits[index]
            
        return self.random_scoring_roll_rec_(scores, biscuit['length']+start)
 
        
    def reset(self):
        self.biscuits = ['_' for i in range(self.size)]
        self.value = 0

    def auto_constraint_rates(self, benefit_rate):
        remain = 1 - benefit_rate
        n_defects = self.getNumbersDefects()
        a = n_defects['a']
        b = n_defects['b']
        c = n_defects['c']
        total = a + b + c
        if total != 0:
            a_rate = (remain*a)/total
            b_rate = (remain*b)/total
            c_rate = (remain*c)/total
            return [benefit_rate, a_rate, b_rate, c_rate]
        return [benefit_rate, 0, 0, 0]

    #Second part, manipulation of partitions of the Roll
    def divide_roll_randomly(self,n_parts, min_size=MINLENGTH):
        """ return index of each partition: array of tuple (start_index, end_index) """
        n_parts = max(n_parts, 1)
        # Calculate the maximum possible size for each part
        max_size = self.size - (n_parts - 1) * min_size
        # Ensure min_size is within the valid range
        min_size = min(min_size, max_size)
        # Generate random indices for splitting the array
        split_indices = sorted(np.random.choice(max_size, n_parts - 1, replace=False))
        # Calculate the actual sizes of the parts
        sizes = np.diff([0] + split_indices + [self.size])
        # Adjust the sizes to meet the minimum size requirement
        sizes = np.maximum(sizes, min_size)
        # Recalculate the split indices based on the adjusted sizes
        split_indices = np.cumsum(sizes[:-1])
        # Calculate start and end indices for each part
        start_indices = [0] + list(split_indices)
        end_indices =  list(split_indices) +  [self.size] 
        partitions_index = []
        return list(zip(start_indices, end_indices))


    def updateFromPartition(self, partition, start):
        self.biscuits[start:start+len(partition.biscuits)] = partition.biscuits.copy()
        self.updateValue()

    def random_partitions_probabilities(self, n_parts, min_size, benefit_rate):
        return 

    #for ABC algorithm
    def updateValue(self):
        self.value = 0
        for i in range(0, self.size):
            if '[' in self.biscuits[i]:
                biscuit = BISCUITS[int(self.biscuits[i][1])]
                self.value += biscuit['value']
                i += biscuit['length']
        
                
    def getRandomPartition(self, max_length):
        """ get a random valid partition from the roll """
        start = random.randint(0, self.size-2)
        if self.biscuits[start] != '_':
            while '[' not in self.biscuits[start]:
                start -= 1
        if start + 1 != self.size-1:
            bound = min(start + max_length -1, self.size-1)
            end = random.randint(start+1, bound)
        else:
            end = self.size-1
            
        if self.biscuits[end] != '_':
            while ']' not in self.biscuits[end]:
                end += 1
        roll_part = self.partition(start, end)
        #copy current biscuits at this position
        roll_part.biscuits = self.biscuits[start:end+1].copy()
        roll_part.updateValue()
        #compute value of this partition
        return roll_part, start, end
        
    def getValidPartition(self, start, end):
        if self.biscuits[start] != '_':
            while '[' not in self.biscuits[start]:
                start -= 1
        if self.biscuits[end] != '_':
            while ']' not in self.biscuits[end]:
                end += 1       
        return self.partition(start, end, copy=True), start, end
        
 
        
    def TournamentPartition(self, min_size):
        """ generates several partitions of the roll, ranks them by they values, and give to them aa probabilitie,
        partition with smaller value are more luck to be choosen"""
        #gererate several partitions of the roll, but each partition can't have just a part of a biscuit
        partitions = []
        step = 0
        while step < self.size-1:
            partition, start, end = self.getValidPartition(step, min(step + min_size - 1,self.size-1))
            step = end + 1
            #score = partition.value/(partition.holl()**2+partition.size)
            score = partition.value/(partition.size)
            partitions.append({'partition' : partition, 'start' : start, 'score' : score})
        #sort the partition in depends of the score
        sorted_partition = sorted(partitions, key=lambda p:p['score'])
        #create probabilities based on the rank of each partition in the sorted array
        ranks = np.arange(1, len(sorted_partition) + 1)
        probabilities = 1 / ranks
        # Normalize probabilities
        probabilities /= probabilities.sum()
        # return the partition to update based on these probabilities
        index = np.random.choice(len(sorted_partition), size=1, p=probabilities)[0]
        return sorted_partition[index]

 
 
            
        
        
        
        

  
        
            
    
        
        


        
            
        
        
        