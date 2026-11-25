#!/usr/bin/env python3
"""Two-layer fighter construction for the Super FX models.

LAYER 1 - ANATOMY.  A per-fighter PHYSIQUE describes the body itself: side-view
width profiles (front / back of the bone, in design units) for the ribcage, pelvis,
thighs, calves, upper arms and forearms, plus joint radii, hand and foot sizes.
Every body segment is built from these profiles, so a heavyweight, a lightweight
and a female fighter get genuinely different bodies, not one template.

LAYER 2 - COSTUME.  Clothing and gear are SLICES of the anatomy profiles (a range
along the bone, a band across it, inflated a little): a boot is the calf profile
from just below the knee to the ankle, puffed out 0.5; a shoulder guard follows
the deltoid dome.  So clothes always sit on the body underneath.

Joint construction (how the pieces join without seams):
  * Every limb starts with a DOME around its proximal joint (hip / shoulder / knee /
    elbow socket).  Rotating the limb rotates the dome about the joint, so the
    silhouette stays continuous through the whole range of motion.
  * The ink outline of a limb starts a little BELOW the joint; the dome above it is
    filled without ink, covering that cut.  A thigh therefore flows out of the pelvis
    and a shin out of the knee, with no outline ring around the joint.
  * Concave anatomy (lumbar curve, calf-to-Achilles taper) is two convex pieces
    whose union makes the curve; a small BRIDGE fill hides the ink where the second
    piece's border lies inside the first.
Draw order: the depth biases in model.export put the near shin over the near thigh,
forearms over upper arms, and the near limbs over the torso, which is what these
seams are designed for.
"""
import math
import model as Mdl
from model import Part, Bone, MAT, SKIN_D, SKIN, SKIN_L, HAIR_D, HAIR, HAIR_L, A_D, A_M, A_L, B_D, B_M, B_L, WHITE, INK

# ------------------------------------------------------------------ physiques
# profiles: (t along the bone, front, back).  t = 0 at the proximal joint, 1 at the
# distal one.  Torso t runs pelvis joint (0) -> chest/shoulder line (1).
PHYSIQUE = {
    # athletic Nordic woman: strong legs, narrow waist, full hips and glutes, slimmer
    # shoulders and arms than the men, small hands and feet
    "VALKA": dict(
        headk=0.86,
        pelvis=[(-0.30, 1.6, 4.4), (-0.20, 4.4, 7.6), (-0.06, 5.8, 9.0), (0.10, 6.2, 8.8), (0.26, 5.8, 6.8), (0.36, 5.4, 5.6)],
        ribs=[(0.28, 5.5, 5.3), (0.38, 5.3, 5.0), (0.52, 6.4, 5.5), (0.66, 9.0, 6.4), (0.80, 9.0, 6.8), (0.92, 7.2, 6.8),
              (1.04, 5.6, 6.0), (1.14, 3.2, 3.6)],
        waist_t=0.37,
        bust=True,
        thigh=[(0.00, 5.4, 6.4), (0.20, 6.6, 6.2), (0.45, 6.2, 5.0), (0.72, 5.0, 4.0), (0.92, 4.2, 3.6), (1.03, 3.9, 3.4)],
        hipball=(5.6, 6.6),
        calf=[(0.00, 4.0, 3.6), (0.14, 4.0, 4.4), (0.30, 3.7, 5.2), (0.46, 3.3, 4.8), (0.62, 2.9, 3.4)],
        ankle=[(0.54, 3.1, 3.9), (0.70, 2.6, 2.7), (0.86, 2.3, 2.2), (0.98, 2.4, 2.5)],
        knee=(4.2, 3.8),
        uarm=[(0.00, 3.9, 4.2), (0.18, 4.0, 3.9), (0.36, 3.4, 3.4), (0.56, 3.6, 3.3), (0.80, 3.1, 2.8), (1.00, 2.6, 2.6), (1.07, 2.5, 2.4)],
        delt=(4.0, 4.3),
        farm=[(0.00, 2.7, 2.9), (0.16, 3.2, 3.1), (0.40, 2.8, 2.6), (0.70, 2.1, 2.0), (0.90, 1.9, 1.8), (0.96, 1.9, 1.8)],
        elbow=(2.6, 3.0),
        fist=(5.0, 2.2, 1.9),          # length, half-height at the knuckles, at the wrist
        foot=0.86,
    ),
}

# ------------------------------------------------------------------ profile maths
def W(st, t):
    """front / back half-widths of a profile at t (clamped to its ends)"""
    if t <= st[0][0]: return st[0][1], st[0][2]
    for (t0, f0, b0), (t1, f1, b1) in zip(st, st[1:]):
        if t <= t1:
            k = (t - t0) / max(1e-6, t1 - t0)
            return f0 + (f1 - f0) * k, b0 + (b1 - b0) * k
    return st[-1][1], st[-1][2]

def span(st): return st[0][0], st[-1][0]

def prof(bone, st, t0=None, t1=None, frac=(0.0, 1.0), inflate=0.0, steps=None):
    """polygon points of a profile slice: t0..t1 along, frac = band across (0 back, 1 front)"""
    a, b = span(st)
    t0 = a if t0 is None else max(a, t0); t1 = b if t1 is None else min(b, t1)
    ts = [t0] + [t for t, _, _ in st if t0 < t < t1] + [t1]
    pts = []
    for t in ts:
        f, bk = W(st, t)
        f += inflate; bk += inflate
        lo = -bk + (f + bk) * frac[0]; hi = -bk + (f + bk) * frac[1]
        pts += [bone.along(t, hi), bone.along(t, lo)]
    return pts

def dome(bone, t, rf, rb, inflate=0.0, n=4):
    """half-disc around the joint at t, bulging toward the parent (-t side)"""
    pts = []
    for i in range(n + 1):
        th = math.pi * i / n                       # 0 = front, pi = back
        c, s = math.cos(th), math.sin(th)
        r = (rf if c >= 0 else rb) + inflate
        pts.append(bone.along(t - r * s / bone.L, r * c))
    return pts

def bridge(p, col, bone, profiles, t, h=1.8, inset=0.9):
    """fill over the junction of two convex pieces at t (hides the inner ink line)"""
    hs = h / bone.L
    def wu(tt):
        ws = [W(st, tt) for st in profiles if span(st)[0] - 1e-6 <= tt <= span(st)[1] + 1e-6]
        return (max(w[0] for w in ws), max(w[1] for w in ws)) if ws else (0, 0)
    f = min(wu(tt)[0] for tt in (t - hs, t, t + hs)) - inset
    b = min(wu(tt)[1] for tt in (t - hs, t, t + hs)) - inset
    if f + b > 0.5:
        p.poly(col, [bone.along(t - hs, f), bone.along(t + hs, f), bone.along(t + hs, -b), bone.along(t - hs, -b)])

def socket_cap(p, col, bone, st, ball, t_cut, inflate=0.0):
    """fill (no ink) from the joint dome down past the outline cut: hides the cut"""
    f, b = W(st, t_cut + 0.10)
    pts = dome(bone, st[0][0], ball[0] + inflate, ball[1] + inflate)
    pts += [bone.along(t_cut + 0.10, f + inflate - 0.8), bone.along(t_cut + 0.10, -(b + inflate) + 0.8)]
    p.poly(col, pts)

def shade(p, mat, bone, st, light=True, t0=None, t1=None, inflate=0.0, dark=0.30):
    """form shading: dark band on the back, lit strip on the front.  On limbs both are
    near-side only (the far variant is already a shade darker and mostly hidden)."""
    d, m, l = MAT[mat] if isinstance(mat, str) else mat
    lim = p.side is not None
    p.poly(d, prof(bone, st, t0, t1, frac=(0.0, dark), inflate=inflate), near_only=lim)
    if light:
        a, b = span(st)
        a = max(a, t0 if t0 is not None else a) + 0.06; b = min(b, t1 if t1 is not None else b) - 0.08
        if b > a: p.poly(l, prof(bone, st, a, b, frac=(0.64, 0.86), inflate=inflate), near_only=lim)

def lerp(a, b, k): return (a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k)

# ------------------------------------------------------------------ the builder
def build(ch, J, style):
    """anatomy + costume parts; head and hair chains come from the legacy builder"""
    PH = PHYSIQUE[ch["name"]]
    legacy = Mdl.build(ch, J, dict(style, headk=PH.get("headk", 1.0), neckblend=True))
    keep = {p.name: p for p in legacy}
    sv = lambda j: (J[j][0], J[j][1])
    M = style["mats"]
    parts = []

    # ================= torso: pelvis + ribcage (anatomy), tunic / guard / belt (costume)
    tb = Bone(sv(0), sv(1))
    tor = Part("torso", 0, 1, tb, depth=0)
    tm = MAT[M["torso"]]
    pel, rib = PH["pelvis"], PH["ribs"]
    tor.poly(tm[1], prof(tb, pel), outline=True)
    tor.poly(tm[1], prof(tb, rib), outline=True)
    bridge(tor, tm[1], tb, [pel, rib], (pel[-1][0] + rib[0][0]) / 2, h=2.6)
    # form shading: back in shadow (glutes, lumbar, upper back), chest lit
    tor.poly(tm[0], prof(tb, pel, frac=(0.0, 0.34)))
    tor.poly(tm[0], prof(tb, rib, t0=0.30, t1=1.06, frac=(0.0, 0.26)))
    tor.poly(tm[2], prof(tb, rib, t0=0.58, t1=0.92, frac=(0.58, 0.84)))
    cos = style.get("costume", {})
    if "guard" in cos:          # chest guard: a plate on the ribcage profile, bust shaped
        gm = MAT[cos["guard"]]
        tor.poly(gm[1], prof(tb, rib, 0.60, 0.98, frac=(0.18, 1.0), inflate=0.5), outline=True)
        tor.poly(gm[0], prof(tb, rib, 0.60, 0.98, frac=(0.18, 0.40), inflate=0.5))
        tor.poly(gm[2], prof(tb, rib, 0.68, 0.86, frac=(0.62, 0.92), inflate=0.5))  # bust highlight
    if "neckline" in cos:       # skin at the collar (the neck sits in it)
        tor.poly(SKIN, prof(tb, rib, 0.97, 1.14, frac=(0.40, 0.92)))
    if "belt" in cos:           # belt at the natural waist
        bm = MAT[cos["belt"]]
        w0 = PH["waist_t"] - 0.07; w1 = PH["waist_t"] + 0.04
        prof_all = sorted(pel + rib)
        tor.poly(bm[1], prof(tb, prof_all, w0, w1, inflate=0.4))
    parts.append(tor)
    parts.append(keep["head"])

    # ================= arms
    sleeve = style.get("sleeve")
    for sh, el, hn, side in ((4, 5, 6, "n"), (7, 8, 9, "f")):
        ub = Bone(sv(sh), sv(el)); fb = Bone(sv(el), sv(hn))
        ua = Part("uarm" + side, sh, el, ub, side=side)
        st = PH["uarm"]; am = "skin"
        ua.poly(MAT[am][1], prof(ub, st, 0.10), outline=True)
        if "pads" not in cos:
            socket_cap(ua, MAT[am][1], ub, st, PH["delt"], 0.10)
        shade(ua, am, ub, st, t0=0.0, light=True)
        if "pads" in cos:       # shoulder guard: follows the deltoid dome
            pm = MAT[cos["pads"]]
            pad = dome(ub, 0.0, PH["delt"][0], PH["delt"][1], inflate=1.4, n=6) + prof(ub, st, 0.34, 0.34, inflate=1.6)
            ua.poly(pm[1], pad, outline=True)
            ua.poly(pm[0], [ub.along(0.0, -PH["delt"][1] - 1.4)] + prof(ub, st, 0.10, 0.34, frac=(0.0, 0.36), inflate=1.6))
            df, db = PH["delt"]
            k = 1.0 / ub.L        # specular crescent on the front-top of the dome
            ua.poly(pm[2], [ub.along(-(db + 0.6) * 0.55 * k, (df + 0.6) * 0.30), ub.along(-(db + 0.9) * 0.82 * k, (df + 0.9) * 0.62),
                            ub.along(-0.2 * k, df + 1.2), ub.along(0.10, df + 0.4), ub.along(0.02, df * 0.45)], near_only=True)
        parts.append(ua)

        fa = Part("farm" + side, el, hn, fb, side=side)
        st = PH["farm"]
        fa.poly(MAT[am][1], prof(fb, st, 0.08), outline=True)
        socket_cap(fa, MAT[am][1], fb, st, PH["elbow"], 0.08)
        shade(fa, am, fb, st, t0=0.0, light=True)
        if "wrist" in cos:      # bracer on the forearm profile
            wm = MAT[cos["wrist"]]
            fa.poly(wm[1], prof(fb, st, 0.56, 0.90, inflate=0.6), outline=True)
            fa.poly(wm[0], prof(fb, st, 0.56, 0.90, frac=(0.0, 0.34), inflate=0.6), near_only=True)
        # fist: proportioned to the head, knuckles forward, fingers curled on the palm side
        ln, hk, hw = PH["fist"]
        k = 1.0 / fb.L; t0 = 0.93
        fist = [fb.along(t0, hw), fb.along(t0 + 1.6 * k, hk + 0.3), fb.along(t0 + (ln - 0.6) * k, hk), fb.along(t0 + ln * k, hk - 1.0),
                fb.along(t0 + ln * k, -hk + 0.6), fb.along(t0 + (ln - 0.8) * k, -hk), fb.along(t0 + 1.4 * k, -hk), fb.along(t0, -hw)]
        fa.poly(SKIN, fist, outline=True)
        fa.poly(SKIN_D, [fb.along(t0 + 1.6 * k, -hk + 1.6), fb.along(t0 + (ln - 0.4) * k, -hk + 1.2),
                         fb.along(t0 + (ln - 0.8) * k, -hk), fb.along(t0 + 1.4 * k, -hk)])          # curled fingers
        if True:
            fa.poly(SKIN_L, [fb.along(t0 + 2.0 * k, hk - 0.2), fb.along(t0 + (ln - 0.8) * k, hk - 0.2), fb.along(t0 + (ln - 1.0) * k, hk - 1.2)], near_only=True)
        parts.append(fa)

    # ================= legs
    for hp, kn, ft, side in ((10, 11, 12, "n"), (13, 14, 15, "f")):
        tbn = Bone(sv(hp), sv(kn)); sb = Bone(sv(kn), sv(ft))
        lm = MAT[M["legs"]]
        th = Part("thigh" + side, hp, kn, tbn, side=side)
        st = PH["thigh"]
        th.poly(lm[1], prof(tbn, st, 0.14), outline=True)
        socket_cap(th, lm[1], tbn, st, PH["hipball"], 0.14)
        shade(th, M["legs"], tbn, st, t0=0.0, light=True)
        th.poly(lm[0], prof(tbn, st, 0.62, 0.86, frac=(0.62, 0.70)), near_only=True)   # quad / knee line
        parts.append(th)

        sh_ = Part("shin" + side, kn, ft, sb, side=side)
        calf, ank = PH["calf"], PH["ankle"]
        # knee: the top of the shin, in leg material, flowing out of the thigh
        boot = cos.get("boots")
        if not boot:
            sh_.poly(lm[1], prof(sb, calf, 0.06, 0.30), outline=True)
        socket_cap(sh_, lm[1], sb, calf, PH["knee"], 0.06)
        if boot:                # boots: the calf and ankle profiles, inflated, from below the knee
            bm = MAT[boot]; bt = cos.get("boot_top", 0.18)
            sh_.poly(bm[1], prof(sb, calf, bt, inflate=0.5), outline=True)
            sh_.poly(bm[1], prof(sb, ank, inflate=0.5), outline=True)
            bridge(sh_, bm[1], sb, [calf, ank], (calf[-1][0] + ank[0][0]) / 2, inset=1.3)
            shade(sh_, boot, sb, calf, t0=bt + 0.06, inflate=0.5, light=True)
            if "fur" in cos:    # fur cuff at the boot top
                fm = MAT[cos["fur"]]
                sh_.poly(fm[1], prof(sb, calf, bt - 0.05, bt + 0.09, inflate=1.2), outline=True)
            fmat = bm
        else:
            sh_.poly(lm[1], prof(sb, calf, 0.2), outline=True)
            sh_.poly(lm[1], prof(sb, ank), outline=True)
            bridge(sh_, lm[1], sb, [calf, ank], (calf[-1][0] + ank[0][0]) / 2)
            shade(sh_, M["legs"], sb, calf, t0=0.2, light=True)
            fmat = MAT[M["feet"]]
        # foot: heel behind the ankle, instep, ball, toe
        F = sv(ft); fw = PH["foot"]
        def Fp(x, y): return (F[0] + x * fw, F[1] + y)
        sh_.poly(fmat[1], [Fp(-2.6, -0.6), Fp(7.2, -0.6), Fp(7.6, 0.6), Fp(6.4, 1.8), Fp(3.4, 3.4), Fp(0.6, 4.8), Fp(-2.2, 4.4), Fp(-3.0, 1.6)], outline=True)
        sh_.poly(fmat[0], [Fp(-2.6, -0.6), Fp(7.2, -0.6), Fp(7.4, 0.4), Fp(-2.9, 0.6)])       # sole
        sh_.poly(fmat[2], [Fp(1.0, 4.0), Fp(3.4, 3.0), Fp(5.8, 1.8), Fp(3.0, 2.2)], near_only=True)  # instep light
        parts.append(sh_)

    parts += [p for p in legacy if p.name.startswith("chain")]
    return parts
