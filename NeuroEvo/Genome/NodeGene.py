

# Creates a node with a nodeNr
class NodeGene:

    def __init__(self, nodeNr, layer = 0, output = False):
        self.nodeNr = nodeNr
        self.layer = layer
        self.output = output
        return

    def copy(self):
        return NodeGene(self.nodeNr, 0, self.output)