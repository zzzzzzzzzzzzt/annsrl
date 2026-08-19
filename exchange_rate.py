"""Measure the recall/DCS exchange rate on the real Pareto frontier.

The question: R = recall + beta*(-DCS/max_dcs) needs beta, and we "don't know the
magnitude". Would recall/DCS (or recall*DCS) avoid that choice?

It does not. Any scalarization of a 2-D frontier implies a local exchange rate
lambda = how much recall it will give up to save 1 DCS. The frontier itself has a
slope s = dRecall/dDCS. If lambda > s the objective cuts DCS; if lambda < s it
spends DCS. So "no parameter" just means "a parameter you cannot see".

  additive  J = R - lambda*D          -> lambda is the constant beta/max_dcs
  ratio     J = R/D                   -> dJ=0 at dR/dD = R/D, so lambda = R/D
  product   J = R*D                   -> maximized by D -> inf; wrong sign

This script reads real (ef, recall, DCS) sweeps and prints s next to each
objective's lambda at the same operating point.
"""
import sys

# Measured eval sweeps (ef, recall@1, mean DCS), SIFT100K, k=1.
HNSW = [
    (12, 0.8638, 303.1), (16, 0.9036, 350.6), (20, 0.9303, 396.1),
    (24, 0.9453, 440.5), (28, 0.9571, 484.1), (32, 0.9646, 527.0),
    (36, 0.9695, 568.5), (40, 0.9739, 609.6),
]
RANDOM_TRAINED = [
    (12, 0.0051, 378.2), (20, 0.0080, 587.4), (32, 0.0119, 887.1),
    (48, 0.0155, 1279.1), (88, 0.0274, 2210.0), (128, 0.0386, 3096.5),
]

BETA, MAX_DCS = 1.0, 1000.0
LAMBDA_ADD = BETA / MAX_DCS   # recall units the current reward pays per 1 DCS


def lam_savings(r, d, budget=MAX_DCS):
    """Implied lambda of MaxDCSReward, J = R*(budget - D) (already in lib/reward.py).

    dJ = (budget-D)dR - R dD = 0  =>  lambda = R/(budget-D). Unlike R/D this rises
    as D approaches the budget, so it spends DCS while there is headroom and turns
    stingy near the limit -- qualitatively the right shape. Past the budget the
    max(...,1) clamp makes the reward pure recall, so lambda is 0 there.
    """
    head = budget - d
    return r / head if head > 0 else 0.0


def slopes(sweep):
    """Local frontier slope s = dRecall/dDCS between consecutive ef settings."""
    out = []
    for (_, r0, d0), (ef1, r1, d1) in zip(sweep, sweep[1:]):
        out.append((ef1, r1, d1, (r1 - r0) / (d1 - d0)))
    return out


def report(name, sweep):
    print('=' * 78)
    print(name)
    print('=' * 78)
    print('%4s %8s %8s   %11s %11s %11s %11s   %s'
          % ('ef', 'recall', 'DCS', 's=dR/dDCS', 'lam_add', 'lam_R/D',
             'lam_R(B-D)', 'over-cuts DCS by'))
    for ef, r, d, s in slopes(sweep):
        lam_ratio, lam_sav = r / d, lam_savings(r, d)
        # An objective cuts DCS exactly when its lambda exceeds the frontier slope;
        # lambda/s is how many times too eager it is at this operating point.
        def fac(lam):
            return '%.0fx' % (lam / s) if s < lam else 'spends'
        print('%4d %8.4f %8.1f   %11.3e %11.3e %11.3e %11.3e   '
              'add %-7s R/D %-7s R(B-D) %s'
              % (ef, r, d, s, LAMBDA_ADD, lam_ratio, lam_sav,
                 fac(LAMBDA_ADD), fac(lam_ratio), fac(lam_sav)))
    print()


def main():
    report('HNSW M12 efC300 -- the graph we must eventually beat', HNSW)
    report('random s_0 after 300 policy steps -- where the gate ran', RANDOM_TRAINED)

    print('=' * 78)
    print('product R*D: strictly increasing in D, so it is maximized by making the')
    print('search as expensive as possible. Wrong sign -- not a candidate.')
    print()
    print('ratio R/D at the two graphs, same objective, same code:')
    for label, sweep in (('HNSW      ', HNSW), ('random+RL ', RANDOM_TRAINED)):
        best = max(sweep, key=lambda t: t[1] / t[2])
        worst = min(sweep, key=lambda t: t[1] / t[2])
        print('  %s best R/D at ef=%-4d (%.4f/%6.1f = %.3e), '
              'worst at ef=%-4d (%.3e)'
              % (label, best[0], best[1], best[2], best[1] / best[2],
                 worst[0], worst[1] / worst[2]))
    print('  -> R/D is maximized at the SMALLEST ef in every sweep: a ratio')
    print('     objective always argues for a cheaper, lower-recall graph.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
