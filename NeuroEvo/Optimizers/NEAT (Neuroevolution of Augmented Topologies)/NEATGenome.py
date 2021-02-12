import Genome.Genome

# Decorator for the Genome class. Wraps around the Genome to give it a score, so it can be used in NEAT.
# Needs to implement a Genome base class.
class NEATGenome(Genome):

    def __init__(self):
        super(NEATGenome, self).__init__()

        return

    # Mutate by adding an edge or node, or tweak a weight
    def mutate(genome):
        return

    # Add an edge connection two nodes
    def addEdge(genome):
        return

    # Replace an edge by a node with the incoming edge having weight 1
    # and the outgoing edge having the original edges weight
    def addNode(genome):
        return

    # Tweak a random weight by adding Gaussian noise
    def tweakWeight(genome):
        return