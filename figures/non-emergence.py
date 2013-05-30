from nips import *

# with a flat prior and softmax = 1, S1 becomes equivalent to old S0 --
# basically just sampling the (renormalized) prior

d = conpact2.MemoizedDomain(max_utterance_length=1, adjectives=2, objects=2,
                            softmax_weight=1)
dialogues = simulate_dialogues(4, 20, d)
show_dialogues("non-emergence-%i.pdf", dialogues)
