

# Connects two nodes by referencing their nodeNrs
class EdgeGene:

    def __init__(self, fromNr, toNr, weight, enabled = True):
        self.fromNr = fromNr
        self.toNr = toNr
        self.weight = weight
        self.enabled = enabled
        return

    def deactivate(self):
        self.enabled = False

    def copy(self):
        return EdgeGene(self.fromNr, self.toNr, self.weight, self.enabled)