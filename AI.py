#!/usr/bin/env python3
""" AI
A Python module implementing greedy search 
and Hill Cimbing solutions for the Travelling Salesman
and Minimum Latency problems. Adapted from:
https://www.python-course.eu/graphs_python.php
"""


class Graph(object):

    def __init__(self, graph_dict=None):
        """ initializes a graph object 
            If no dictionary or None is given, 
            an empty dictionary will be used.
            The dictionary is supposed to have vertex keys
            and for each key k the value is another dictionary
            whose keys are the sucessor vertices v of k and
            which possibly has as values a third dictionary having a
            "weight" key, giving the weight of the edge (k, v), a "label"
            key, giving the label of the edge. These last two fields are optional.
        """
        if graph_dict == None:
            graph_dict = {}
        self.__graph_dict = graph_dict
        self.__dimension = len(graph_dict)

    @classmethod
    def complete_graph(cls, n, weights=None):
        """ Generates a complete graph for n nodes.
            The weighs value is an n by n matrix of floating point numbers 
            and if none is specified it will default to one containing 0s.
        """
        d = {}
        if weights == None:
            weights = [[0 for j in range(n)] for i in range(n)]
        for k in range(n):
            d[k] = {v: {"weight": weights[k][v], "label": ""} for v in range(n) if not v == k}
        return cls(d)

    def dimension(self):
        """ returns the dimension of the graph (i.e. number of vertices) """
        return self.__dimension

    def vertices(self):
        """ returns the vertices of a graph """
        return list(self.__graph_dict.keys())
    
    def dict(self):
        """ retuns a dictionary representation of the graph """
        return self.__graph_dict

    def edges(self):
        """ returns the edges of a graph """
        return self.__generate_edges()

    def edgeWeight(self, i, j):
        """ returns the weight of edge (i, j) """
        return self.__graph_dict[i][j]["weight"]
    
    def edgeLabel(self, i, j):
        """ returns the label of edge (i, j) """
        return self.__graph_dict[i][j]["label"]

    def add_vertex(self, vertex):
        """ If the vertex "vertex" is not in 
            self.__graph_dict, a key "vertex" with an empty
            dictionary as a value is added to the dictionary. 
            Otherwise nothing has to be done. 
        """
        if vertex not in self.__graph_dict:
            self.__graph_dict[vertex] = {}

    def add_edge(self, edge, weight = 0.0, label = ""):
        """ assumes that edge is a tuple(pair) of two vertex values;
        """
        (vertex1, vertex2) = tuple(edge)
        if vertex1 in self.__graph_dict:
            self.__graph_dict[vertex1][vertex2] = {"weight": weight, "label": label}
        else:
            self.__graph_dict[vertex1] = {vertex2: {"weight": weight, "label": label}}

    def __generate_edges(self):
        """ A private method generating the edges of the 
            graph "graph". Edges are represented as pairs 
            of vertices
        """
        edges = []
        for vertex in self.__graph_dict:
            for neighbour in self.__graph_dict[vertex]:
                if (vertex, neighbour) not in edges:
                    edges.append((vertex, neighbour))
        return edges

    def __str__(self):
        res = "vertices: "
        for k in self.__graph_dict:
            res += str(k) + " "
        res += "\nedges: "
        for edge in self.__generate_edges():
            res += str(edge) + " "
        return res

    def objective(self, factor):
        """ returns a generic objective function for paths in the graph relative
            to some choice of multiplicative factor, which is a list of values 
            to be multiplied in sequence by the weights of edges of the graph.
            The function is then given by the sum of all such terms.
            Because of the lazyness of the iterative type and generator expressions,
            the returned objective function will have O(len(path)) time complexity.
        """
        return lambda path: sum(self.edgeWeight(path[i], path[i+1]) * factor[i] for i in range(len(path) - 1))


def tsp(graph):
    """ Computes the objective function (heuristic) for the
        Traveling Salesman Problem.
    """
    return lambda path: sum(graph.edgeWeight(path[i], path[i+1]) for i in range(len(path) - 1))

def mlt(graph):
    """ Computes the objective function (heuristic) for the
        Minimum Latency Problem.
    """
    return lambda path: sum(graph.edgeWeight(path[i], path[i+1]) * (len(path) - 1 - i) for i in range(len(path) - 1))

def next(graph, path, objective=None, maximize=False):
    """ Computes the next greedy choice of vertex for a complete graph
        to go to for the partial solution of a hamiltonian path and returns the next
        solution containing the chosen vertex. The "objective" parameter gives
        the objective function which calculates total cost for a given solution,
        given in the parameter "path". 
    """
    if objective == None:
        objective = tsp(graph)

    #function to compare vertices to choose
    comp = lambda v: objective(path + [v])
    #dictionary of graph
    d = graph.dict()    
    #notice the first argument to sorted is the list of vertices acessible to the
    #last last vertex of the path which are not in the path.
    #we do this in one line so that we can guarantee the usage of a lazy generator for
    #this list of vertice instead of relying in a non-lazy list comprehension.
    bestVertex = sorted((v for v in d[path[-1]] if not v in path), key=comp, reverse = maximize)[0]
    return path + [bestVertex]

def greedyCircuit(graph, initialVertex=0, path=[], maximize=False, objective=None):
    """ Computes a hamiltonian circuit by a greedy algorithm following an heuristic.
        It utilizes the "next" function to calculate the next path from the current
        one and, in the end, appends the initial vertex to the path. By default it 
        uses the tsp heuristic and seeks to minimize it. Notice that if you dont
        specify an initialVertex and the graph doesn't contain a "0" vertex, this
        function will throw an error.
    """
    if objective == None:
        objective = tsp(graph)
    if not path:
        path = [initialVertex]
    if len(path) == graph.dimension():
        return path + [initialVertex]
    newPath =  next(graph, path, objective, maximize)
    return greedyCircuit(graph, initialVertex, newPath, maximize, objective)

""" Hill Climbing Implementation """

def swapSpace(circuit):
    """ Generate the neighborhood space for a
        hamiltonian circuit which is based on
        swapping 2 elements of the path.
        Considering that the first and last
        nodes of the path remain the same for
        all elements of the space, there are in total
        n**2 - 5*n + 6 other such circuits in the space,
        where n is the length of the circuit.
        This function is also a generator.
    """
    for i in range(1, len(circuit) - 1):
        for j in range(i+1, len(circuit) - 1):
            solution = circuit.copy()
            solution[i], solution[j] = solution[j], solution[i]
            yield solution

# auxiliary functions
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    """ Auxiliary function for float comparison,
        which would be unecessary as of Python 3.5.
        Copied from 
https://stackoverflow.com/questions/5595425/what-is-the-best-way-to-compare-floats-for-almost-equality-in-python
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

fst = lambda pair: pair[0]

def hillClimb(circuit, graph, constructor=swapSpace, objective=None, maximize=False):
    """ Generates a hamiltonian circuit by best fit Hill Climbing upon
        a previous solution. By default it uses the tsp heuristic
        and seeks to minimize it.
    """
    if objective == None:
        objective = tsp(graph)
    space = [(objective(solution), solution) for solution in constructor(circuit)]
    space = sorted(space, key=fst, reverse=maximize)
    result = space[0]
    value = objective(circuit)
    cond = value > result[0] if maximize else value < result[0]
    if isclose(value, result[0]) or cond:
        return circuit
    return hillClimb(result[1], graph, constructor, objective, maximize)

def main(n):
    weights = [[i+j for j in range(n)] for i in range(n)]
    graph = Graph.complete_graph(n, weights)
    objective=mlt(graph)
    solution = greedyCircuit(graph, objective=objective)
    print("greedy solution: " + str(solution))
    print("value: " + str(objective(solution)))
    solution = hillClimb(solution, graph, objective=objective)
    print("\nhill climbed solution: " + str(solution))
    print("value: " + str(objective(solution)))
    

if __name__ == '__main__':
    main(58)



