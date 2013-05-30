import multiprocessing
import conpact2
import cPickle
import numpy as np

SOFTMAX_WEIGHT = 3
LISTENER_DEPTH = 1
PARTICLE_COUNT = 1000

def simulate_dialogue(r, dialogue, turns):
    for turn in xrange(turns):
        obj, utt, interp, backchannel = dialogue.sample_obj_utt_interp_backchannel(r)
        # Speaker observes listener's interpretation; listener observe
        # speaker's intended meaning
        s_datum = conpact2.SDataInterp(listener_depth, utt, interp)
        l_datum = conpact2.LDataUttWithMeaning(listener_speaker_depth,
                                               utt, obj)
        dialogue.new_data([s_datum], [l_datum])

def horn_dialogue(i_turns):
    i, turns = i_turns
    bgl_domain = conpact2.MemoizedDomain(adjectives=2, objects=2,
                                         max_utterance_length=1,
                                         softmax_weight=SOFTMAX_WEIGHT,
                                         word_cost=[0.5, 1.0],
                                         default_object_prior=[0.8, 0.2])
    r = np.random.RandomState(i)
    flat = np.log([0.5, 0.5])
    def trace_horn_S_lexicalization(turn, dialogue):
        return np.exp(dialogue.uncertain_s_dist(log_object_prior=flat))
    def trace_horn_L_lexicalization(turn, dialogue):
        return np.exp(dialogue.uncertain_l_dist(log_object_prior=flat))

    tracers = {"P-understood": conpact2.trace_P_understood,
               "horn-S-lexicalization": trace_horn_S_lexicalization,
               "horn-L-lexicalization": trace_horn_L_lexicalization,
               }
    dialogue = conpact2.Dialogue(domain, listener_depth,
                                 i, PARTICLE_COUNT,
                                 tracers=tracers)
    simulate_dialogue(r, dialogue, turns)

    return i, turns, dialogue.traces

def par_horn_dialogues(count, turns, outpath):
    p = multiprocessing.Pool()
    jobs = [(i, turns) for i in xrange(count)]
    for (i, turns, traces) in p.imap_unordered(horn_dialogue, jobs):
        cPickle.dump((i, turns, traces),
                     open("%s/horn-%s-%s.pickle" % (outpath, turns, i), "w"),
                     protocol=-1)
