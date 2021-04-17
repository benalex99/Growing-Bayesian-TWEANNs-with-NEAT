from NeuroEvo.Genome.ConnectionGene import EdgeGene

class AdvancedEdgeGene(EdgeGene):
    def __init__(self, fromNr, toNr, weight, enabled = True, hMarker = 0, toClass = None, fromClass = None):
        super(AdvancedEdgeGene,self).__init__(fromNr, toNr, weight, enabled, hMarker)
        self.toClass = toClass
        self.fromClass = fromClass
        return

    def deactivate(self):
        self.enabled = False

    def __deepcopy__(self, memodict={}):
        return AdvancedEdgeGene(self.fromNr, self.toNr, self.weight, self.enabled, self.hMarker, self.innerInput, self.innerOutput)

    def __repr__(self):
        return str(self.fromNr) + " " + str(self.toNr) + " " + str(self.weight) + " " + str(self.enabled)