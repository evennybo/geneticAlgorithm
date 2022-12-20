from logging.config import valid_ident
import math
import igraph as ig
import matplotlib.pyplot as plt
import random

#This class will create an adjmetrics list that contais all the conneted nodes 
class Graph(object):
    # Initialize the matrix
    def __init__(self, size):
        self.adj_matrix = []
        for i in range(size):
            self.adj_matrix.append([0 for i in range(size)])
        self.size = size

    # Add edges to the graph
    def add_edge(self, row, column):
        if row == column:
            print("Same vertex %d and %d" % (row, column))
        self.adj_matrix[row][column] = 1
        self.adj_matrix[column][row] = 1

    # Remove edges from the graph
    def remove_edge(self, row, column):
        if self.adj_matrix[row][column] == 0:
            print("No edge between %d and %d" % (row, column))
            return
        self.adj_matrix[row][column] = 0
        self.adj_matrix[column][row] = 0

    # Returns size of matrix
    def __len__(self):
        return self.size
    # Print the matrix
    def print_matrix(self):
        for row in self.adj_matrix:
            for val in row:
                print('{:4}'.format(val))
            print("\n")
    # Returns the matrix 
    def get_matrix(self):
        return self.adj_matrix

#This class implements all of the elements that is needed to generate an genetic algorithm 
class GeneticAlgorithm(Graph):
    
    def __init__(self, graph, population_size):
        self.graph = graph
        self.population_size = population_size
        best_path = []

      
# Generates a random chromoson by using the random number generator provided in python
# Adds a a random length for the chromosone within the range og 5-15 nodes 
# It also checks if the chromosone nodes are following the rules of the graph links
    def generate_chromosome(self):
        population_size = random.randint(5, 15)
        chromosome = [random.randrange(18)]
        while True:
            node = []
            for i in range (0, self.graph.size):
                if self.graph.adj_matrix[chromosome[-1]][i] != 0: #Checks if the nodes are connected
                    node.append(i) # Appends the linked nodes to the chromosone
            next_node = random.choice(node)
            chromosome.append(next_node)
            #Checks if the length of the chromosone is the same as the population size for then to return the new chromosone
            if len(chromosome) == population_size: 
                break      
        return chromosome

                      
    #Generates a certain amount of chromosome dependent on the population size that has been decided
    def initialize_population(self):
        population = []
        for i in range(self.population_size):
            population.append(self.generate_chromosome()) #appends new chromosones until it matches the size of population
        return population

  
    #Finds the length of every chromosone and returns the cost of path
    def cost_fitness(self, c ):
        cost = 0
        cost  = len(c)
        return cost

 
    #Checks how many nodes a spesific chromosone has and returns the number of
    # nodes user wants to visit that appears in that spesific chromosone
    def cost_nodes(self, c, nodes):# check the cost at the begining
        score = 0
        index_nodes = []
        for i in range(0,len(nodes)):
            if nodes[i] in c:
               score += 1 
               index_nodes.append(c.index(nodes[i]))
        return score, index_nodes #Returns the score of the node and the indexes where the nodes appear in chromosone

    #Returns the score of chromosones, for each node that appears in that spesific chromosomes
    # it adds the a 100 value, if all nodes is found in the chromosone it will have a score of 400
    # it also adds the length of the chromosone so it will be possible to see the diffrence between chromosones 
    # that has the same amount of nodes
    def scores(self, population, nodes):
        scores = []
        for i in range(0,len(population)):
            #Checks if the chromsone has one or more nodes
            if self.cost_nodes(population[i], nodes)[0] > 0:
                sum = 0
                chrom = population[i]
                sum = sum + len(chrom) #adds the leght to the sum
                sum = sum + (self.cost_nodes(population[i], nodes)[0]*100) #add appearence of nodes in chromosone
                scores.append(sum)
            else: 
                sum = 0
                chrom = population[i]
                sum = sum + len(chrom) #adds only the length of the chromosone without nodes
                scores.append(sum)

        return scores #Return the scores

  
    #Validates if the link of a two nodes are linked in the matrix
    #Returns True if it has a link or false if no link has been found
    def graphValidation(self, chrom1, crosspoint, matrix):
        valid = False
        coordinate_1 = chrom1[crosspoint]
        coordinate_2 = chrom1[crosspoint + 1]
        if matrix[int(coordinate_1)][int(coordinate_2)] != 0:  #Checks if the coordinates are valid links
            valid = True
        return valid 

  

# Crosses the best chromoson from the last population with the rest of the population
    def crossover(self, chrom1,  population, nodes, matrix):
        crosspoint = 0
        new_gen = []
        new_gen.append(chrom1)
        for i in range(0,len(population)):
            if len(chrom1) > len(population):
                crosspoint = random.randint(1,len(population))
            elif len(chrom1) < len(population):
                crosspoint = random.randint(1,len(chrom1))
            #Creates two empty list to appending new strong chromosons
            new_chrom1= []
            new_chrom2 = []
            #Split the chromosome and create the new chromosome 1
            crosspoint = 2
            for i in range(0,len(population)):
                gen = population[i]
                gen = gen[crosspoint:]
                chrom1_crop = chrom1[:crosspoint]
                new_chrom1 = chrom1_crop + gen
            #validates if the new crossed chromsone has a valid link
            check_1 = self.graphValidation(new_chrom1, crosspoint, matrix) 
            
            if check_1 == False: ######
                for i in range(0, len(population)):
                    new_chrom1.append(population[i])

            #Split the chromosome and create the new chromosome 2
            for i in range(0,len(population)):
                gen = population[i]
                gen = gen[:crosspoint]
                chrom2_crop = chrom1[crosspoint:]
                new_chrom2 = gen + chrom2_crop
            #validates if the second new crossed chromsone has a valid link
            check_2 = self.graphValidation(new_chrom2, crosspoint, matrix)
            #if the validation is false it will keep the old chromsone to the population
            if check_2 == False: 
                for i in range(0, len(population)):
                    new_chrom2.append(population[i])
            new_gen.append(new_chrom1) #if true it will append to the new population
            new_gen.append(new_chrom2) #if true it will append to the new population
        return new_gen


    def mutation(self, chromosome, prob):
            new_chromosome = chromosome[:]
            for i in range(1, len(new_chromosome)-1):
                if random.random() < prob:
                    options = []
                    for j in range(self.graph.size):
                        if self.graph.adj_matrix[new_chromosome[i-1]][j] != 0 and j != i:
                            options.append(j)
                        for node in options:
                            pos = -1
                            for j in range(1, len(new_chromosome)-1):
                                if new_chromosome[j] == node:
                                    pos = j
                                    break

                            check_1 = self.graph.adj_matrix[new_chromosome[i]][new_chromosome[pos+1]]
                            check_2 = self.graph.adj_matrix[new_chromosome[pos]][new_chromosome[i+1]]
                            check_3 = self.graph.adj_matrix[new_chromosome[pos-1]][new_chromosome[i]]

                            if pos != -1 and check_1 and check_2 and check_3:
                                new_chromosome[pos], new_chromosome[i] = new_chromosome[i], new_chromosome[pos]
                                break

            return new_chromosome

## Checks which node that has one node or more in population and kills every solution who doesn't have a
    def strongest_population(self, population, nodes):
        strongest_population = []
        scores = self.scores(population, nodes)
        #Kills all the chromosome nodes that does not contain any nodes
        for i in range(0,len(population)):
            if scores[i] > 100:
                strongest_population.append(population[i])
        #If the population does not decrease since the first selection, 
        # it will look for all the chromosones that has two or more nodes in them        
        #Kills all the chromosome nodes that contains less than two nodes
        if len(strongest_population) == len(population) or len(strongest_population) >= len(population) - 2: #Cuts down population
            strongest_population = []
            for i in range(0,len(population)):
                if scores[i] > 200:
                    strongest_population.append(population[i])
        #If the population does not decrease since the first selection, 
        # it will look for all the chromosones that has three or more nodes in them        
        #Kills all the chromosome nodes that contains less than three nodes
        if len(strongest_population) == len(population) or len(strongest_population) >= len(population) - 2: #Cuts down populations
            strongest_population = []
            for i in range(0,len(population)):
                if scores[i] > 300:
                    strongest_population.append(population[i])

        #If the population does not decrease since the first selection, 
        # it will look for all the chromosones that has all nodes in them        
        #Kills all the chromosome nodes that is not a successfull path
        if len(strongest_population) == len(population): #Cuts down population
            strongest_population = []
            for i in range(0,len(population)):
                if scores[i] > 400:
                    strongest_population.append(population[i])

        return strongest_population


# finds the bast path in populations and retrun the best path
    def selection(self, population, nodes):
        new_gen = []
        scores = self.scores(population, nodes)
        best = max(scores)
        index_best  = scores.index(best)
        #Takes the best score and returns the best path
        difference = lambda scores : abs(scores - 400)
        best = min(scores, key = difference)
        index_best  = scores.index(best)
        new_gen = population[index_best]
        new_pop = []
        #Shortens the population by getting the path that includes nodes
        new_pop = self.strongest_population(population, nodes)
        #Adds the strongest chromoson from the population
        new_pop.append(new_gen)
        return new_pop, new_gen

   
    # Finds the start node from a path and shrinks the oath cost becasue it has all the Nodes visited
    # Only shinks when a path has been found, andfind the most optimal path by taking away unneccecary nodes.
    def shrink(self, c, nodes):
        new_c = []
        location = self.cost_nodes(c, nodes)[1]
        if self.cost_nodes(c, nodes)[0] == len(nodes):
            for j in range(min(location), max(location)+1):
                new_c.append(c[j])
        return new_c

    #Checks if a path is a successfull path or not
    # Returns true if a ptah is a successfull path, otherwise returns false
    def valid_path(self, population, nodes):
        valid_paths = []
        valid = 0
        scores = self.scores(population, nodes) #get scores 
        for i in range(0,len(population)):
            if scores[i] > 400: #checks if all nodes are in chromosone
                valid_paths.append(population[i]) #appends it to a list if all nodes is there
                valid += 1 
        if len(valid_paths) == 0:
            return False
        return True


def main():
    x = []
    y = []
    nodes=[]
    f = open("hw3_cost239.txt")
    next(f)
    next(f)
    for row in f:
        row = row.split(' ')
        x.append(int(row[0]))
        y.append(int(row[1]))

    a = math.ceil(len(x)/2)
    g = Graph(a)

    for i in range(len(x)):
        g.add_edge(x[i], y[i])
    for i in range(4):
        number = int(input("Enter a number between 0-18 as a node to find a path through: "))
        nodes.append(number)

    graph = ig.Graph.Adjacency(g.get_matrix(), "undirected")
    ga = GeneticAlgorithm(g,10)     
    matrix = g.get_matrix()
    graph.vs["num"] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]

    fig, ax = plt.subplots(figsize=(5,5))
    #displays the graph
    ig.plot(

        graph,

        target=ax,

        layout="kk", # print nodes in a circular layout

        vertex_size=0.2,

        vertex_color=["steelblue" if num in nodes else "lightgray" for num in graph.vs["num"]],

        vertex_frame_width=1.0,

        vertex_frame_color="black",

        vertex_label=graph.vs["num"],

        vertex_label_size=10.0,

        edge_width=[1]

    )
    plt.show()

    generations = 10
    finalsolutions = []
    #has a while loop that decides the amount of generations it will run
    while generations != 0:
        new_pop = ga.initialize_population()  #creates a new population
        solutions = []
        for i in range(0,6): #goes for the amont of the generations
            best_gen = ga.selection(new_pop, nodes)[1]
            new_pop = ga.selection(new_pop, nodes)[0]
            solutions.append(best_gen)
            #does the crossover after first generation has been selected
            crossover = ga.crossover(best_gen, new_pop, nodes, matrix)
            best_gen = ga.selection(crossover, nodes)[1]
            new_pop = ga.selection(crossover, nodes)[0]


            solutions.append(best_gen) #adds to solution list

        #mutation is not getting called due to a bug in the mutation function
            #mutation = ga.mutation(new_pop, 0.05)
            #best_gen = ga.selection(mutation, nodes)[1]
            #new_pop = ga.selection(mutation, nodes)[0]
            solutions.append(best_gen)

        best_path = ga.selection(solutions, nodes)[1] 
        shrunk = ga.shrink(best_path, nodes)
        finalsolutions.append(shrunk) #final solution 

        #if there is a solution, print the graph again with solution
        if len(shrunk) !=0:
            #print how many generations and the path found at that generation 
              print("Took this many generations to find a solution:", generations)
              print('Path found: ', shrunk)
              
              final = ig.Graph.Adjacency(g.get_matrix(), "undirected")
              #the _within edge sequence selector selects all edges
              # that can connect nodes within the shrunken solution.
              #This does not perfectly display the path, but the 
              # path is included in every one.
              #The color of the nodes is changed to blue for all nodes in the path as well to 
              #further demonstrate where the path should be.
              final.es["color"] = "black"
              red_edges = final.es.select(_within=(shrunk))
              red_edges["color"] = "red"
              final.vs["num"] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
              # Plot solution graph
              fig, ax = plt.subplots(figsize=(5,5))
              #displays the graph
              ig.plot(
                     final,
                     target=ax,
                     layout="kk", # print nodes in a circular layout
                     vertex_size=0.2,
                     vertex_color=["steelblue" if num in shrunk else "lightgray" for num in final.vs["num"]],
                     vertex_frame_width=1.0,
                     vertex_frame_color="black",
                     vertex_label=final.vs["num"],
                     vertex_label_size=10.0,
                     edge_width=[1]
              )
              plt.show()
        else:
              print("No path found.")
  
        generations -= 1

    #print the best solution found
    print()
    print()
   
    #runs selection function on finalsolutions list to pick the smallest one

    a = ga.selection(finalsolutions, nodes)[1]
    if a != []:
        print("This is the best path found: ", a)
    else:
        print("No paths were found.")


if __name__ == '__main__':

    main()

 