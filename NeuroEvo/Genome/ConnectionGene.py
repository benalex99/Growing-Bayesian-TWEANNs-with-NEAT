

# Connects two nodes by referencing their nodeNrs
class EdgeGene:

    def __init__(self, fromNr, toNr, weight):
        self.fromNr = fromNr
        self.toNr = toNr
        self.weight = weight
        self.enabled = True
        return

    def deactivate(self):
        self.enabled = False