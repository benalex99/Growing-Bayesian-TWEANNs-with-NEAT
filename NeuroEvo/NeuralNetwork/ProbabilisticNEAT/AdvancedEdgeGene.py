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
        return AdvancedEdgeGene(self.fromNr, self.toNr, self.weight, self.enabled, self.hMarker, self.toClass, self.fromClass)

    def __repr__(self):
        return str(self.fromNr) + " " + str(self.toNr) + " " + str(self.weight) + " " + str(self.enabled)

    def toData(self):
        return [self.fromNr, self.toNr, self.weight, self.enabled, self.hMarker, self.toClass, self.fromClass]

    @staticmethod
    def fromData(data):
        return AdvancedEdgeGene(data[0], data[1], data[2], data[3], data[4], data[5], data[6])
