

# Creates a node with a nodeNr
class NodeGene:

    def __init__(self, nodeNr, layer = 0, output = False, input = False, outputtingTo = []):
        self.nodeNr = nodeNr
        self.layer = layer
        self.output = output
        self.input = input
        self.outputtingTo = outputtingTo
        return

    def copy(self):
        outputtingTo = []
        for x in self.outputtingTo:
            outputtingTo.append(x)
        return NodeGene(self.nodeNr, self.layer, self.output, self.input, outputtingTo= outputtingTo)