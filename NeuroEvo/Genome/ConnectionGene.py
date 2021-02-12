

# Connects two nodes by referencing their nodeNrs
class EdgeGene:

    def __init__(self, fromNr, toNr, weight):
        self.fromNr = fromNr
        self.toNr = toNr
        self.weight = weight
        return