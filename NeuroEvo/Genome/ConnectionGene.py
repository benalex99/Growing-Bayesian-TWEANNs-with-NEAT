

# Connects two nodes by referencing their nodeNrs
class EdgeGene:

    def __init__(self, fromNr, toNr, weight, enabled = True, hMarker = 0):
        self.fromNr = fromNr
        self.toNr = toNr
        self.weight = weight
        self.enabled = enabled
        self.hMarker = hMarker
        return

    def deactivate(self):
        self.enabled = False

    def copy(self):
        return EdgeGene(self.fromNr, self.toNr, self.weight, enabled = self.enabled, hMarker = self.hMarker)

    def __repr__(self):
        return str(self.fromNr) + " " + str(self.toNr) + " " + str(self.enabled)