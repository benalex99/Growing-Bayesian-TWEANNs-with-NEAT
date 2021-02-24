# First networkx library is imported  
# along with matplotlib 
import networkx as nx
import matplotlib.pyplot as plt
#import matplotlib.transforms.Bbox as Bbox

# Defining a Class 
class Visualizer:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a 
        # graph
        print("new")
        self.G = nx.DiGraph()


    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b):
        self.G.add_edge(a,b)

    def addNode(self, nodeNr, pos = (0,0)):
        self.G.add_node(nodeNr, pos = pos)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list 
    # nx.draw_networkx(G) - plots the graph 
    # plt.show() - displays the graph 
    def visualize(self, ion= True):
        print("vis")
        plt.cla()
        if(ion):
            if (not plt.isinteractive()):
                plt.ion()
        else:
            if (plt.isinteractive()):
                plt.ioff()
        pos = nx.get_node_attributes(self.G, 'pos')

        nx.draw(self.G, pos)
        plt.show()
        plt.pause(0.001)
        print("done")