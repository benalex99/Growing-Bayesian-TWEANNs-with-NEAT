

# Creates a node with a nodeNr
class NodeGene:

    def __init__(self, nodeNr, layer = 0, output = False, input = False, outputtingTo = None):
        if outputtingTo is None:
            outputtingTo = []
        self.nodeNr = nodeNr
        self.layer = layer
        self.output = output
        self.input = input
        if(outputtingTo == None):
            self.outputtingTo = []
        else:
            self.outputtingTo = outputtingTo
        return

    def __deepcopy__(self, memodict={}):
        return NodeGene(self.nodeNr, self.layer, self.output, self.input, outputtingTo= self.outputtingTo.copy())

    def __repr__(self):
        if(self.output):
            nodeType = "Output"
        elif(self.input):
            nodeType = "Input"
        else:
            nodeType = "Hidden"

        return "NodeNr: " + str(self.nodeNr) \
               + " Layer: " + str(self.layer) \
               + " Outputs to: " \
               + str(self.outputtingTo)  + " "\
               + "NodeType: " + nodeType
