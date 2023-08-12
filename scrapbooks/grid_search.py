#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 20:20:02 2023

@author: sheehabpranto
"""


# Grid Search to select optimal hyper Parameter

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Parameters for grid search
POP_SIZES = [20, 50, 70, 100, 150, 200, 300]
GENS = [15, 18, 25, 30, 35, 45]
CROSSOVER_PROBS = [0.75, 0.85, 0.95, 0.99]
MUTATION_RATE = 0.2

best_accuracy = 0
best_params = {}

for pop_size in POP_SIZES:
    for gens in GENS:
        for crossover_prob in CROSSOVER_PROBS:

            # Initialize population with random max_depth values between 1 and 10
            population = np.random.randint(1, 11, pop_size)

            for generation in range(gens):
                # Evaluate fitness
                fitness = []
                for individual in population:
                    clf = DecisionTreeClassifier(max_depth=individual)
                    clf.fit(X_train, y_train)
                    val_pred = clf.predict(X_val)
                    fitness.append(accuracy_score(y_val, val_pred))

                # Select parents (top 50%)
                parents = population[np.argsort(fitness)[-pop_size//2:]]

                # Create next generation using crossover and mutation
                next_population = []
                for i in range(0, pop_size, 2):
                    parent1 = parents[i % (pop_size//2)]
                    parent2 = parents[(i + 1) % (pop_size//2)]

                    # Crossover
                    if np.random.rand() < crossover_prob:
                        child1 = (parent1 + parent2) // 2
                        child2 = (parent1 + parent2) // 2
                    else:
                        child1, child2 = parent1, parent2

                    # Mutation
                    if np.random.rand() < MUTATION_RATE:
                        child1 += np.random.randint(-2, 3)
                        child1 = max(1, child1)
                    if np.random.rand() < MUTATION_RATE:
                        child2 += np.random.randint(-2, 3)
                        child2 = max(1, child2)

                    next_population.extend([child1, child2])

                population = np.array(next_population)

            # Evaluate the best solution on the validation set
            best_individual = population[np.argmax(fitness)]
            clf = DecisionTreeClassifier(max_depth=best_individual)
            clf.fit(X_train, y_train)
            val_pred = clf.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)

            # Update best parameters if necessary
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = {
                    'pop_size': pop_size,
                    'gens': gens,
                    'crossover_prob': crossover_prob,
                    'best_max_depth': best_individual
                }

print(f"Best Parameters: {best_params}")
print(f"Best Validation Accuracy: {best_accuracy}")

# Evaluate on test set using best parameters
clf = DecisionTreeClassifier(max_depth=best_params['best_max_depth'])
clf.fit(X_train, y_train)
test_pred = clf.predict(X_test)
print(f"Test Accuracy: {accuracy_score(y_test, test_pred)}")