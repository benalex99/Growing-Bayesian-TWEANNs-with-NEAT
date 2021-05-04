import ast

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
        for i in range(len(data)):
            if data[i] == '':
                data[i] = 'None'
        return AdvancedEdgeGene(ast.literal_eval(data[0]),
                                ast.literal_eval(data[1]),
                                ast.literal_eval(data[2]),
                                ast.literal_eval(data[3]),
                                ast.literal_eval(data[4]),
                                ast.literal_eval(data[5]),
                                ast.literal_eval(data[6]))
