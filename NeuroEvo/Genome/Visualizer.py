# First networkx library is imported  
# along with matplotlib 
import matplotlib.pyplot as plt
import networkx as nx


#import matplotlib.transforms.Bbox as Bbox

# Defining a Class 
class Visualizer:

    def __init__(self):
        # visual is a list which stores all
        # the set of edges that constitutes a 
        # graph
        # self.G = nx.DiGraph()
        self.G = nx.MultiDiGraph()



    # addEdge function inputs the vertices of an
    # edge and appends it to the visual list
    def addEdge(self, a, b, rad=0.0):
        self.G.add_edge(a,b, rad=rad)

    def addNode(self, nodeNr, pos = (0,0), size = 0):
        self.G.add_node(nodeNr, pos = pos, size = size)

    # In visualize function G is an object of
    # class Graph given by networkx G.add_edges_from(visual)
    # creates a graph with a given list 
    # nx.draw_networkx(G) - plots the graph 
    # plt.show() - displays the graph 
    def visualize(self, ion= True, labels = None, edgeLabels=None, edgeRads=None):
        plt.cla()
        if(ion):
            if (not plt.isinteractive()):
                plt.ion()
        else:
            if (plt.isinteractive()):
                plt.ioff()
        pos = nx.get_node_attributes(self.G, 'pos')

        if labels == None:
            nx.draw_networkx_nodes(self.G, pos)
        else:
            nx.draw_networkx_nodes(self.G, pos)

        nx.draw_networkx_labels(self.G, pos, labels=labels)

        # nx.draw_networkx_edges(self.G, pos) #, connectionstyle='arc3, rad = 0.0')
        for edge in self.G.edges(data=True):
            nx.draw_networkx_edges(self.G, pos, edgelist=[(edge[0],edge[1])], connectionstyle=f'arc3, rad = {edge[2]["rad"]}')


        if edgeLabels != None:
            # edge_labels = dict([((n1, n2), f'{n1}->{n2}')
            #                     for n1, n2 in self.G.edges])
            text = nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edgeLabels, font_size=8, alpha=0.5)
            for _, t in text.items():
                t.set_rotation('horizontal')
        plt.show()
        plt.pause(0.001)