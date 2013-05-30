from nips import *

# Horn implicature alone

# A trivial version of Leon's common/rare cheap/expensive domain
bgl_domain = dom(adjectives=2, objects=2,
                 word_cost=[0.5, 1.0],
                 default_object_prior=[0.8, 0.2])
dialogue = conpact2.Dialogue(bgl_domain, LISTENER_DEPTH, 0, PARTICLES)

print "How speaker refers to \"common\" and \"rare\" objects:"
print np.exp(dialogue.uncertain_s_dist())
print "How listener interprets \"cheap\" and \"expensive\" words:"
print np.exp(dialogue.uncertain_l_dist())

# Emergence + Horn implicature

# XX TODO: calculate some statistics on P(convergence to good system) by
# running a few hundred dialogues
#
# XX mention: this is a very general result, can speculate about it as a
# partial cause of things like common words being shorter, rise of useful
# social conventions in general
#
# XX mention: Horn implicature can't survive as implicatures. And connect to
# scalar implicature, which *can* survive.

bgl_dialogues = simulate_dialogues(4, 20, bgl_domain)
show_dialogues("horn-emergence-%i.pdf", bgl_dialogues)
