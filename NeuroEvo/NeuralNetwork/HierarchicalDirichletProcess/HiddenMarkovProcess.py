
# predict state based on observation
# predict state based on previous state and observation

# Assume the exchangeability of our data is unknown. (It may be exchangeable) How do we determine if it is?
# Can we find a "probability" that it is exchangeable?

# Assuming data is not exchangable. How much data do we take in to predict a state?
# If we have one batch, we have only one state.
# We can model any amount of states by varying the batchsizes.
# We can model batches by clustering data together.
# We can generate arbitrary clusters with the dirichlet process.


# take a child that is just born. It will receive sensory input at an unknown rate.
# The child must somehow infer which sensory input belongs to a single "situation".
# A situation can take split seconds up to hours, days or even months.

# When we see an object, when do we determine that we have perceived the object?
# We may have an internal feeling of when we are certain enough what object we have in front of us.
# However, the perception of an object is arguably infinite. We can inspect an object from infinitely many angles and my find infinitely small details.

# Objects may reveal more features over time.
# The question is, like with situations, when do we cut off our observation and assume we know all there to know.

# This is where arbitrary clustering comes in.
# We may attempt to model the worlds workings by segmenting it into objects and defining relations between these objects.
# What objects exist depends on our segmentation of percepts into groups of observations that belong to the same thing.
# A dirichlet process allows us to create arbitrarily many clusters with arbitrarily many data points.