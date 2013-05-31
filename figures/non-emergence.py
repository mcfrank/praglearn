from nips import *

from scipy.special import gammaln

# two agents each have dirichlet priors on a set of object -> word
# multinomial distributions.
# without pragmatics we can't condition on listener interpretation, so
# instead, they take turns speaking

SIZE = 2

def DM_dist(alpha):
    # Joint density for N samples from a dirichlet-multinomial
    # distribution with parameters alpha[i], and n_k
    # repeats of each outcome is
    #   Z * prod_k Gamma(n_k + alpha[k])
    # where:
    #   Z = Gamma(A) / Gamma(N + A) * prod_k 1/Gamma(alpha[k])
    #   A = sum_k(alpha[k])
    #   N = sum_k(n_k)
    # For us N = 1
    # So basically P(i|alpha) \propto gamma(1 + alpha[i])/gamma(alpha[i])
    dist = gammaln(1 + alpha) - gammaln(alpha)
    dist -= np.logaddexp.reduce(dist)
    return dist

def marginal_S_dist(posterior):
    dist = np.empty((SIZE, SIZE))
    for obj in xrange(SIZE):
        dist[:, obj] = DM_dist(posterior[:, obj])
    return dist

def L_dist(posterior):


def simulate_non_pragmatic(r, turns, prior=None):
    speaker_posterior = np.ones((SIZE, SIZE))
    listener_posterior = np.ones((SIZE, SIZE))

    for i in xrange(turns):
        target = r.randint(SIZE)
        S_dist = marginal_S_dist(speaker_posterior)
        L_dist = marginal_S_dist(listener_posterior)
        L_dist -= np.logaddexp.reduce(L_dist, axis=1)[:, np.newaxis]
        assert np.allclose(np.logaddexp.reduce(S_dist, axis=0), 1)
        assert np.allclose(np.logaddexp.reduce(L_dist, axis=1), 1)


    # Want out:
    #   P(understood)
    #   P(obj|word) using both posteriors
