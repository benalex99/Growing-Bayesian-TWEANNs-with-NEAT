

# Creates a node with a nodeNr
class NodeGene:

    def __init__(self, nodeNr, layer = 0):
        self.nodeNr = nodeNr
        self.layer = layer
        return

    def copy(self):
        return NodeGene(self.nodeNr, self.layer)