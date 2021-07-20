import numpy as np
from sklearn.metrics import accuracy_score

# load the validation predictions
c1 = np.load()
c2 = np.load()
c3 = np.load()
c4 = np.load()
c5 = np.load()
c6 = np.load()

# load the validation ground-truth
y_val = np.load()

# change NUMBER_OF_CLASSIFIERS and CLASSIFIERS_PREDICTIONS to the number of classifiers and the c1, c2, c3 ...
NUMBER_OF_CLASSIFIERS = 5
CLASSIFIERS_PREDICTION = [c1, c2, c3, c4, c5]
CLASSIFICATION = 8
WEIGHTS = NUMBER_OF_CLASSIFIERS * CLASSIFICATION
EARLY_STOPPING = 50
EPOCHS = 1000



def normalize_pop(pop):
  return (pop.T / np.max(pop, axis=1)).T



def init(pop, weights):
  pop_size = (pop, weights)
  new_population = np.random.uniform(low = 0.0, high = 1.0, size = pop_size)
  new_population = normalize_pop(new_population)
  return new_population



def fitness(pop):
  metric = []

  for individual in pop:
    result = np.zeros(8)
    for idx_classifier in range(NUMBER_OF_CLASSIFIERS):
      result = result + np.multiply(CLASSIFIERS_PREDICTION[idx_classifier], individual[idx_classifier * CLASSIFICATION:(idx_classifier + 1) * CLASSIFICATION])
    result = np.argmax(np.array(result), axis = 1)
    metric.append(accuracy_score(y_val, result))

  return metric



def select_parents(pop, fitness, num_parents):

  parents = np.empty((num_parents, pop.shape[1]))
  for parent_num in range(num_parents):
    max_fitness_id = np.where(fitness == np.max(fitness))[0][0]
    parents[parent_num, :] = pop[max_fitness_id, :]
    fitness[max_fitness_id] = -99999999999

  return parents



def crossover(parents, offspring_size):
  offspring = np.empty(offspring_size)

  for i in range(900):
    a, b = np.random.choice(np.arange(100), 2, replace=False)

    p1 = parents[a]
    p2 = parents[b]

    for j in range(WEIGHTS):
      prob = np.random.uniform(low = 0.0, high = 1.0)
      offspring[i][j] = (p1[j] * prob) + (p2[j] * (1 - prob))
  offspring = normalize_pop(offspring)
  return offspring



def mutation(offspring, offspring_size):
  offspring_mut = np.empty(offspring_size)

  for idx_individual in range(len(offspring)):
    for idx_weight in range(WEIGHTS):
      random_value = np.random.normal(loc=0, scale=1)
      if random_value > 0:
        offspring_mut[idx_individual][idx_weight] = offspring[idx_individual][idx_weight] * (1 + random_value)
      else:
        offspring_mut[idx_individual][idx_weight] = offspring[idx_individual][idx_weight] / (1 - random_value)
  
  offspring_mut = normalize_pop(offspring_mut)
  return offspring_mut





population = init(2000, WEIGHTS)

best_individual = None
best_fitness = -np.inf
early_stopping = EARLY_STOPPING

for generation in range(EPOCHS):
  print("Generation: ", generation + 1)
  fit = fitness(population)
  best_match_idx = np.where(fit == np.max(fit))[0][0]
  
  #Early stopping
  if best_fitness >= fit[best_match_idx]:
    if early_stopping == 0:
      print('Early Stopping')
      break
    else:
      early_stopping -= 1
  else:
    early_stopping = EARLY_STOPPING
  
  best_individual = population[best_match_idx]
  best_fitness = fit[best_match_idx]
  print('Best Fitness:', best_fitness)
  print()

  parents = select_parents(population, fit, 100)
  offspring_crossover = crossover(parents, offspring_size=(900, WEIGHTS))
  offspring_mutation = mutation(np.concatenate((parents, offspring_crossover)), offspring_size=(1000, WEIGHTS))

  population[:100, :] = parents[:, :]
  population[100:1000, :] = offspring_crossover[:, :]
  population[1000:, :] = offspring_mutation[:, :]



fit = fitness(population)
local_population = select_parents(population, fit, 200)

best_individual = None
best_fitness = -np.inf
early_stopping = EARLY_STOPPING

for generation in range(EPOCHS):
  print("Generation: ", generation + 1)
  fit = fitness(local_population)
  best_match_idx = np.where(fit == np.max(fit))[0][0]
  
  #Early stopping
  if best_fitness >= fit[best_match_idx]:
    if early_stopping == 0:
      print('Early Stopping')
      break
    else:
      early_stopping -= 1
  else:
    early_stopping = EARLY_STOPPING

  best_individual = local_population[best_match_idx]
  best_fitness = fit[best_match_idx]
  print('Best Fitness:', best_fitness)
  print()
    
  parents = select_parents(local_population, fit, 100)
  offspring_mutation = mutation(parents, offspring_size=(100, WEIGHTS))
    
  local_population[:100, :] = parents[:, :]
  local_population[100:, :] = offspring_mutation[:, :]


print(best_individual)


def eval_fitness(individual):
  result = np.zeros(8)
  for idx_classifier in range(NUMBER_OF_CLASSIFIERS):
    result = result + np.multiply(CLASSIFIERS_PREDICTION[idx_classifier], individual[idx_classifier * CLASSIFICATION:(idx_classifier + 1) * CLASSIFICATION])
  return result

result = eval_fitness(best_individual)
np.save('validation-prediction.npy', result)






# load the test predictions
c1 = np.load()
c2 = np.load()
c3 = np.load()
c4 = np.load()
c5 = np.load()
c6 = np.load()

# change NUMBER_OF_CLASSIFIERS and CLASSIFIERS_PREDICTIONS to the number of classifiers and the c1, c2, c3 ...
NUMBER_OF_CLASSIFIERS = 5
CLASSIFIERS_PREDICTION = [c1, c2, c3, c4, c5]

result = eval_fitness(best_individual)
np.save('test-prediction.npy', result)