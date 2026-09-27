from .poses import *  # noqa: F401,F403 — cascades the full upstream namespace,

# mirroring the flat file's definition order exactly (each module sees
# everything defined before it, helpers included).

# ------------------------------------------------------------- headstand
# Inverted-pose trick on the full-collision scene. Recipe lessons applied
# preemptively: every positive term is two-layer or wide (gradient at the
# CURRENT behavior, i.e. standing upright), the "doing it" pay is per-step and
# state-gated (no jackpots), anchors mild during discovery, motion-blockers
# LOW (this needs a dynamic pitch-over the calm taxes would smother).

def _head_bodies(env):
    ids = getattr(env, "_head_contact_bodies", None)
    if ids is None:
        ids = {mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")}
        env._head_contact_bodies = ids
    return ids


def _head_on_floor(env) -> bool:
    # Consulted by up to four backflip/headstand terms per step; memoized in
    # the step-scoped cache (walk_env) so the contact scan runs once. The old
    # loop also re-materialized data.contact.geom1/geom2 arrays through the
    # bindings on EVERY contact row — hoisted, like _foot_contacts.
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("head_floor")
        if v is not None:
            return v
    heads = _head_bodies(env)
    v = False
    n = int(env.data.ncon)
    if n:
        con = env.data.contact
        g1 = con.geom1.tolist()
        g2 = con.geom2.tolist()
        floor = env.floor_geom
        bodyid = env.model.geom_bodyid
        for i in range(n):
            a = g1[i]
            b = g2[i]
            if a == floor:
                other = b
            elif b == floor:
                other = a
            else:
                continue
            if int(bodyid[other]) in heads:
                v = True
                break
    if cache is not None:
        cache["head_floor"] = v
    return v


def _feet_body_ids(env):
    ids = getattr(env, "_feet_body_ids", None)
    if ids is None:
        ids = {mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)
               for n in ("ankle_left", "ankle_right")}
        env._feet_body_ids = ids
    return ids


def _body_floor_contacts(env) -> int:
    """Floor contacts from anything that ISN'T the head or the feet — i.e.,
    the trunk/hips/legs dragging on the ground. A 'headstand' that reads
    perfect on orientation can still be a chest-slump (trunk z 0.033 with 584
    trunk-floor contacts, caught by the user's eye): orientation and
    ELEVATION are independent, and both must be priced."""
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("body_floor")
        if v is not None:
            return v
    heads, feet = _head_bodies(env), _feet_body_ids(env)
    v = 0
    n = int(env.data.ncon)
    if n:
        con = env.data.contact
        g1 = con.geom1.tolist()
        g2 = con.geom2.tolist()
        floor = env.floor_geom
        bodyid = env.model.geom_bodyid
        for i in range(n):
            a = g1[i]
            b = g2[i]
            if a == floor:
                other = b
            elif b == floor:
                other = a
            else:
                continue
            bid = int(bodyid[other])
            if bid not in heads and bid not in feet:
                v += 1
    if cache is not None:
        cache["body_floor"] = v
    return v


def _inverted(env) -> float:
    """Two-layer upside-down-ness: gravity_z in the trunk frame is -1 upright,
    +1 in a headstand. Wide layer pays ~0.13 even fully upright — the slope
    that makes pitching over discoverable at all."""
    gz = float(env._projected_gravity()[2])
    d2 = (gz - 1.0) ** 2  # 0 when inverted, 4 when upright
    return 0.6 * float(np.exp(-d2 / 1.4 ** 2)) + 0.4 * float(np.exp(-d2 / 0.35 ** 2))


def _jaw_bid(env) -> int:
    """jaw_soft's body id, resolved by name ONCE per env (a per-step
    mj_name2id lookup measured ~2 us in three different terms)."""
    bid = getattr(env, "_jaw_body_id", None)
    if bid is None:
        bid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft")
        env._jaw_body_id = bid
    return bid


def _head_low(env) -> float:
    """Head shell near the floor (target measured off the model: resting jaw
    center ~6 cm; standing is 23 cm — wide std keeps the slope alive).

    UNWIRED: no term reads this. It was a shepherd in the five-potential era;
    headstand_hold now judges the whole stack, which subsumes it. Kept as a
    building block — measured against the real model, so it is worth more
    than a re-derivation — but it costs nothing until something wires it."""
    z = float(env.data.xpos[_jaw_bid(env)][2])
    return float(np.exp(-((z - 0.06) ** 2) / 0.12 ** 2))


def _feet_up(env) -> float:
    """Feet high overhead. Target 30 cm (head base ~6 + trunk + EXTENDED legs)
    — the original 18 cm target was reachable by a crumpled tuck, and the
    policy delivered exactly that crumple. Wide std keeps the slope alive from
    tuck height. Front-side + inversion gated: a back-of-head rest with
    feet in the air must not collect this."""
    g = env._projected_gravity()
    if float(g[0]) < -0.12:
        return 0.0  # backbend rest — same gate as the hold
    if _hs_too_rolled(env):
        return 0.0  # side-prop kickstand — see _body_lifted's roll gate
    gz = float(g[2])
    gate = max(0.0, min(1.0, gz / 0.7))
    if gate == 0.0:
        return 0.0
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0  # a planted-foot bow must not collect this
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    z = min(zl, zr)  # both feet, not one waving
    return gate * float(np.exp(-((z - 0.30) ** 2) / 0.12 ** 2))


def _legs_straight_up(env) -> float:
    """Straight, extended legs — but ONLY while inverted (gated on gravity)
    and with the feet OFF the floor (a feet-planted plank collected this at
    full pay): the standing start keeps its bent STAND pose unpunished. Raw
    joint angles toward zero = a straight vertical line, like the reference
    headstand."""
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    g = env._projected_gravity()
    if float(g[0]) < -0.12:
        return 0.0  # backbend rest — same gate as the hold
    if _hs_too_rolled(env):
        return 0.0  # side-prop kickstand — see _body_lifted's roll gate
    gz = float(g[2])
    gate = max(0.0, min(1.0, gz / 0.7))
    if gate == 0.0:
        return 0.0
    q = env._joint_qpos()[C.LEG_JOINT_IDS]
    d2 = float((q * q).sum())
    return gate * (0.5 * float(np.exp(-d2 / 2.0 ** 2))
                   + 0.5 * float(np.exp(-d2 / 0.7 ** 2)))


def _gentle_plant_pen(env) -> float:
    """One-time charge at the instant the head TOUCHES DOWN, scaled by impact
    speed² (<= 0). Serves two goals at once: the plant itself must be gentle
    (camera + ToF live in that shell), and every RETRY costs its own plant —
    so a single clean first-try attempt beats flip-drop-reflip cycling."""
    head_now = _head_on_floor(env)
    prev = getattr(env, "_gp_prev_head", False)
    env._gp_prev_head = head_now
    if not head_now or prev:
        return 0.0
    _, _, vz = _base_vel(env)
    # Flat fee + impact scale — MODERATE on purpose. History: impact-only →
    # free dabbing; flat-heavy → fast plants; then 0.6+20v² (at weight 2)
    # crossed the attempt-tax line and the policy simply STOPPED doing the
    # trick from upright (gorgeous 0.15 m/s taps, zero headstands). Entry
    # must stay clearly profitable; this pricing nudges toward few, soft
    # touches without ever making "don't try" the argmax.
    return -(0.3 + 10.0 * vz * vz)


def _calm_inverted_pen(env) -> float:
    """Stillness while inverted (<= 0). The flip-in (gz < 0.5) stays untaxed;
    once it is up on the head, thrash charges, so catching beats rolling past
    and flailing. Without this the recipe paid a flip-teeter-drop as well as a
    real hold. Depth is the ONLY gate — see the note below on why a penalty
    must not carry the salaries' pose gates."""
    g = env._projected_gravity()
    # NO front-side or roll gate here, deliberately — unlike its five sibling
    # SALARIES. This is a penalty: gating it withholds the CHARGE, so every
    # gated region becomes a zero-cost refuge beside an honest crown balance
    # that IS charged for its corrections. (The front-side gate that used to
    # sit here cited an overshoot terminal that was removed the same day, and
    # made the whole backbend region free to thrash in.) Gates belong on pay,
    # not on price.
    gz = float(g[2])
    gate = max(0.0, min(1.0, (gz - 0.5) / 0.4))
    if gate == 0.0:
        return 0.0
    w = env._gyro
    return gate * -0.06 * float(w[0] ** 2 + w[1] ** 2 + w[2] ** 2)


def _body_lifted(env) -> float:
    """Trunk ELEVATED to the clean-stack height while inverted (gated) —
    two-layer toward the measured spawn-pose trunk z of ~0.165. The missing
    complement to orientation: pays for head-body-feet actually stacking.
    Feet must be OFF the floor (caught in review: a plank with feet planted
    collected this salary at full trunk height). Head must be planted and
    the trunk itself off the floor — otherwise a hovering fall or a
    chest-slump with legs still up would collect the tallness annuity."""
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    if not _head_on_floor(env):
        return 0.0
    if _body_floor_contacts(env) > 0:
        return 0.0
    g = env._projected_gravity()
    if float(g[0]) < -0.12:
        return 0.0  # backbend rest — same gate as the hold
    # Roll gate: every other gate lives in the pitch plane, so a 37° SIDE
    # lean on the head-shell edge — statically stable, zero balance skill —
    # collected +6.9/step across the salaries (measured 2026-09-01, the
    # kickstand the whole fleet parked in). The gz/0.7 gate saturates at
    # 0.7 and the lean has gz 0.75; only g[1] can see it.
    if _hs_too_rolled(env):
        return 0.0
    gz = float(g[2])
    gate = max(0.0, min(1.0, gz / 0.7))
    if gate == 0.0:
        return 0.0
    z = float(env._trunk_xpos[2])
    d2 = (z - 0.165) ** 2
    return gate * (0.5 * float(np.exp(-d2 / 0.10 ** 2))
                   + 0.5 * float(np.exp(-d2 / 0.04 ** 2)))


def _body_drag_pen(env) -> float:
    """Penalty (<= 0) whenever the trunk/hips/legs touch the floor — the
    direct price on chest-slumping (head and feet contacts stay free)."""
    return -1.0 if _body_floor_contacts(env) > 0 else 0.0


def _nose_down(env) -> float:
    """Direction shaping: the trick is a FRONT flip — face-plant first, then
    legs over the head. Pays for leaning nose-DOWN (gravity acquiring +x in
    the trunk frame). gz is symmetric and can't tell a front headstand from a
    backbend, so without this the policy went over backwards (user caught it
    on video comparison).

    UNWIRED: no term reads this. The wrong_way PENALTY took over the job of
    stopping the backwards entry — a charge for going the wrong way, rather
    than pay for going the right way, so there is no direction annuity to
    farm. Kept because the lesson it records outlives the term."""
    return max(0.0, min(1.0, float(env._projected_gravity()[0])))


def _wrong_way_pen(env) -> float:
    """Penalty (<= 0) for the backbend route: gravity tipping toward -x
    (head craning back under the body). Zero while upright or nose-down."""
    gx = float(env._projected_gravity()[0])
    return -2.0 * max(0.0, -gx - 0.15)


_FACE_FLOOR_MATS = (
    "face_part", "noenoeil", "lens_material", "jaw_material",
    "jaw_soft_material", "bottom_head_shell", "soft_mouth",
)


def _face_on_floor_pen(env) -> float:
    """-1 when the FACE/jaw/underside is on the floor, 0 when only the
    crown (top_head_shell) is. The hold's contact test sees one body
    (jaw_soft) so a face-lean and a crown-roll look the same; this is
    the geom-level 'on the TOP of the head' price. The 152° crown spawn
    contacts only top_head_shell and is not charged."""
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("face_floor")
        if v is not None:
            return v
    v = 0.0
    n = int(env.data.ncon)
    if n:
        con = env.data.contact
        g1, g2 = con.geom1, con.geom2
        floor = env.floor_geom
        matid = env.model.geom_matid
        for i in range(n):
            a, b = int(g1[i]), int(g2[i])
            other = b if a == floor else a if b == floor else None
            if other is None:
                continue
            mid = int(matid[other])
            if mid < 0:
                continue
            mat = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, mid) or ""
            if any(s in mat for s in _FACE_FLOOR_MATS):
                v = -1.0
                break
    if cache is not None:
        cache["face_floor"] = v
    return v


# Chin-to-chest so the CROWN (top_head_shell) meets the floor, not the face.
# neck_pitch and head_pitch have opposite world axes (neck body is 180° about
# X in the MJCF; the viewer's look-down rig is neck−, head+). Same-sign
# targets (−0.6, −0.6) extend the neck and plant the FACE — measured on the
# inverted spawn: face/eye 5 mm *below* the crown. Opposite signs roll the
# contact onto the skull (face 4 cm above the crown).
_HS_NECK_TUCK = -0.8
_HS_HEAD_TUCK = 0.6
# sin(30°) = 0.50. A 37° side-prop kickstand is ~0.60 and must stay unpaid;
# a BAM catch wobbles 20–25° (|g[1]| ≈ 0.34–0.42) and must still collect.
# The first gate (0.35 ≈ 20°) zeroed the honest stack and training stalled.
_HS_ROLL_MAX = 0.50


def _hs_too_rolled(env) -> bool:
    return abs(float(env._projected_gravity()[1])) > _HS_ROLL_MAX


def _neck_tuck(env) -> float:
    """Chin-to-chest while inverted (gated on gravity). The reference
    technique (real microduck footage): face-plant, TUCK the neck — which
    rolls the contact point from the beak to the crown and pulls the pivot
    under the spine — then kick over. Untucked, the duck cranes its head back,
    rests on its face, and vertical alignment is geometrically impossible."""
    g = env._projected_gravity()
    if float(g[0]) < -0.12:
        return 0.0  # backbend rest — same gate as the hold
    if _hs_too_rolled(env):
        return 0.0
    gz = float(g[2])
    gate = max(0.0, min(1.0, gz / 0.7))
    if gate == 0.0:
        return 0.0
    q = env._joint_qpos()
    d2 = float((q[5] - _HS_NECK_TUCK) ** 2 + (q[6] - _HS_HEAD_TUCK) ** 2)
    return gate * (0.5 * float(np.exp(-d2 / 0.8 ** 2))
                   + 0.5 * float(np.exp(-d2 / 0.3 ** 2)))


def _feet_on_top(env) -> float:
    """User-specified stack salary (2026-08-31): BOTH feet airborne and
    carried above the head AND the trunk — legs on top of the stack, not
    flopped beside it. Graded on the LOWER foot so one leg up doesn't pay;
    inversion-gated and front-side-only so a neck-bridge with legs in the
    air (an exploit spotted on screen) collects nothing. Priced
    off the measured poses: straight stack ~1.0, tucked hold ~0.09,
    flop-lean ~0 — a direct, heavy salary for exactly the missing piece."""
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    # Same base gates as the hold — this is a salary for legs on top OF A
    # HEADSTAND. Without them the first student to meet this term lay flat
    # on its trunk, cranked its neck into an S so the head tipped down, and
    # poked its feet up — 1.5/step for napping (980cd9-s1, watch sheet).
    if not _head_on_floor(env):
        return 0.0
    if _body_floor_contacts(env) > 0:
        return 0.0
    g = env._projected_gravity()
    if float(g[0]) < -0.12:
        return 0.0  # backbend/neck-bridge rest doesn't count
    if _hs_too_rolled(env):
        return 0.0  # side-prop kickstand — the gate every sibling salary has
    gz = float(g[2])
    gate = max(0.0, min(1.0, (gz - 0.3) / 0.4))
    if gate == 0.0:
        return 0.0
    head_z = float(env.data.xpos[_jaw_bid(env)][2])
    trunk_z = float(env._trunk_xpos[2])
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    lo = min(zl, zr)
    above_head = max(0.0, min(1.0, (lo - head_z - 0.05) / 0.12))
    above_trunk = max(0.0, min(1.0, (lo - trunk_z) / 0.08))
    return gate * above_head * above_trunk


# Late CATALOG additions (registered here because they reuse the headstand's
# gates, defined above the base catalog): benched from the default recipe but
# offered as "+ add a term" sliders in the teach panel — the repo's home for
# documented alternatives instead of dead code. calm_up_top is the polish
# term (it taxes the correction thrash a learner needs — add it only to
# fine-tune an existing hold); feet_on_top is the heavy stack salary variant.
CATALOG["calm_up_top"] = RewardTerm(
    "calm_up_top",
    "Penalty for thrashing while balanced on the head (polish — taxes learning)",
    1.0, _calm_inverted_pen, is_penalty=True)
CATALOG["feet_on_top"] = RewardTerm(
    "feet_on_top",
    "Points for carrying both feet above the head and the body",
    2.0, _feet_on_top)


def _headstand_hold_raw(env) -> float:
    """The per-step trick pay: head planted + both feet airborne → pays STEEPLY
    with full inversion. The first version paid `gz` from a 0.3 gate, and the
    policy parked in a comfy 60° face-bow collecting most of every term — the
    playbook's compromise basin, verbatim. The 0.5 gate that replaced it still
    paid 43% at a 146° face-lean (measured 2026-08-31 while chasing the
    parked-dive exploit), so the gate now sits at 0.85: pay begins ~148°,
    a 155° beak-bow earns ~8%, and only crown-vertical collects real money.
    The head-contact test can't tell beak from crown (the whole head is one
    body, jaw_soft) — this gate is what enforces "on the TOP of the head"."""
    if not _head_on_floor(env):
        return 0.0
    if _body_floor_contacts(env) > 0:
        return 0.0  # ONLY the head may touch — a chest-slump earns nothing
    c = env.foot_contact_state
    if c["left"] or c["right"]:
        return 0.0
    g = env._projected_gravity()
    if float(g[0]) < -0.12:
        return 0.0  # backbend rest doesn't count — front-flip entries only
    if _hs_too_rolled(env):
        return 0.0  # 37° side-prop kickstand
    gz = float(g[2])
    # Strictness is STAGED via env knobs (defaults = full strictness). The
    # crown-only gate and stack factor correctly kill the C-basin late, but
    # they starve ignition early: a from-scratch learner lives in wobbly
    # ~155° semi-holds that the old 0.5 gate paid ~2.6/step and this gate
    # pays ~0.17/step — 15x less, and no scratch brain ignited under it
    # (513090/10baca stage 1 flat at hold ~0.05 even on xml servos, vs the
    # permissive-era scratch run's 1.3@3M). Training wheels loosen the
    # gate; the drill stages restore full pricing, which provably
    # straightens a hold once balance exists (849641-s1: 1.29 -> 2.67).
    # Default 0.5, NOT 0.85: with the ordering factor below constitutional,
    # a permissive depth gate no longer reopens the C-hold exploit (wrongly
    # ordered poses pay zero at any depth) — and a strict gate leaves the
    # measured -145° dive-trap with no gradient out (the parallel-session
    # diagnosis, 2026-08-31 night: "make the hold kick in BEFORE the parked
    # pose"). Correct order + modest depth pays; the park never does.
    gate = float(_spawn_knob(env, "MICRODUCK_HS_GATE", "0.5"))
    x = max(0.0, (gz - gate) / (1.0 - gate))
    base = x ** 2
    # The STACK factor (user-specified success test, 2026-08-31): a real
    # headstand is head on the ground, body ABOVE the head, feet ABOVE that.
    # Orientation alone let a low TUCK — trunk near-vertical, feet dangling
    # at head height — collect full pay. Reference heights: head ~0.035,
    # clean-stack trunk ~0.165 (_body_lifted's target), extended feet ~0.30
    # (_feet_up's target). A tuck keeps ~1/3 of base pay (the hold skill
    # still matters); the full salary demands the full stack.
    head_z = float(env.data.xpos[_jaw_bid(env)][2])
    trunk_z = float(env._trunk_xpos[2])
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    lo_feet = min(zl, zr)
    # THE ORDERING IS CONSTITUTIONAL (the maintainer's definition, stated three
    # times before it finally became a hard rule): head lowest, body above
    # the head, feet above the body — in EVERY stage. Only inversion DEPTH
    # is laddered (the gate knob above); shape is never negotiable.
    # Ramps reach DOWN to the measured parks (2026-09-01, run 8a8ce3): the
    # capped-std graduate balances an inverted tuck-ball — crown moments at
    # trunk-head gap ~0.017 and feet ~0.14, both just BELOW the old ramp
    # starts (0.02 / 0.18), a zero-gradient plateau (same disease as the
    # -145° dive-trap: the pay must kick in before the parked pose). Pay
    # from gap 0 / feet 0.12 so every centimeter of extension out of the
    # ball earns more; the full stack still pays ~10x the ball.
    s_body = max(0.0, min(1.0, (trunk_z - head_z) / 0.06))
    # 4 cm of feet-above-trunk let a FOLDED PIKE collect full hold (ba4c43
    # @13M: hold 3.6/step, legs_straight 0.07) — the toe-press camp, body
    # mass still behind the head. Demand real extension: the lower foot
    # toward the straight-stack 0.30 m, so a pike at ~0.22 m keeps ~1/3.
    # CONVEX in feet height: slope is alive from 0.12 (the tuck-ball can
    # always earn more by unfolding a little) but squared so the toe-press
    # pike at ~0.22 still pays only ~30% — the pike camp lock (ba4c43 @13M)
    # holds while the plateau below it is gone.
    s_feet = max(0.0, min(1.0, (lo_feet - 0.12) / 0.18)) ** 2
    return base * s_body * s_feet


def _headstand_hold(env) -> float:
    """The raw hold pay scaled by a PERSISTENCE ramp: transits pay 30%, and a
    full second of continuous qualifying dwell reaches 100%. Exists because
    two straight capped-std graduates (5935be, 443c65) converged to the same
    deterministic behavior: survive all 8 s inverted but FLICKER through the
    strict pose ~0.3 s at a time — under flat per-step pay a rocking limit
    cycle through the stack collects almost as much as dwelling in it, and is
    dynamically far easier than stabilizing the unstable equilibrium. The
    ramp makes dwelling strictly richer while adding NOTHING farmable: it
    only scales pay that already passed every gate, and every parked pose
    still earns zero. Floor 0.3, not 0: a scratch learner's first wobbly
    transits must still pay (the strict-gate starvation lesson) — effective
    transit weight 8.0*0.3 = 2.4, comparable to the permissive-era 4.0-hold
    that provably ignited. Streak bookkeeping lives in _hs_update (one call
    per step); this term stays pure so probes and previews can price poses."""
    raw = _headstand_hold_raw(env)
    streak = getattr(env, "_hs_streak", 0)
    return raw * (0.3 + 0.7 * min(streak / 50.0, 1.0))


# Progress-pay bookkeeping (Behavior.state_fn). Three straight graduates
# (840b8a, its fine-tune, 6e880c with slump drills) all converged to the
# same move from standing: dive to ~146°, then FREEZE — because the pose
# terms below paid ~2.9/step for merely BEING partially inverted, forever,
# motionless. "A term a policy can farm will get farmed" (AGENTS.md) — a
# per-step state pay is an annuity, and parking was a salaried job. These
# terms now pay only the INCREASE over the episode's best-so-far: a full
# dive still earns every point of its slope exactly once, a parked duck's
# shaping income is zero within a step, and the only per-step salary left
# is the hold itself — which demands head-only contact, both feet in the
# air, and near-crown-vertical inversion. Baselines initialize from the
# FIRST post-spawn state, so reverse-curriculum handout spawns bank no
# progress money either: they are paid in hold, or not at all.


# body_lifted is deliberately NOT here: it went through the progress-pay
# conversion and came back out. As a one-time delta the flop-lean hold
# (trunk ~0.065, legs collapsed sideways) costs nothing — and stage-1
# brains converged to exactly that (feet_up flat 0.13, spotted
# the flopped legs on screen). As a per-step salary it is the TALLNESS
# incentive the champion's lineage learned straight legs under, and it is
# not farmable lying down: inversion-gated and targeting the full-stack
# trunk height 0.165, a parked heap collects ~0 of it.
def _hs_feet_height(env) -> float:
    """Potential for the feet-rise shepherd: the LOWER foot's height,
    normalized by the extended-stack target (0.30 m). min() so both feet
    must rise — one leg waved in the air pays half a stack, not a full one.
    Exists because the four-rule covenant's first graduate parked in the
    perfect PREP (crown planted at -166°, feet still on the floor behind —
    f2b99e, watch sheet): the upside-down shepherd was banked, the hold
    only pays after the feet unload, and nothing paid the leap between."""
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    return min(1.0, min(zl, zr) / 0.30)


# EXACTLY the potentials some registered term reads back through
# _hs_gain_term. head_low, nose_down and neck_tuck used to sit here too, left
# over from the five-shepherd era: _hs_update evaluated all five every step
# and the reward read two, so three were pure hot-loop cost. They also read as
# live terms — the anti-parking test summed "head_low" and "nose_down" out of
# this tuple for a reward key that never existed, scoring 0.0 forever and
# quietly covering less than it claimed. Keep this tuple and the registered
# gain terms in one-to-one correspondence.
_HS_POTENTIALS = (("upside_down", _inverted), ("feet_rise", _hs_feet_height))


def _hs_update(env) -> None:
    # SYMMETRIC potential-based shaping (Ng et al.), not best-so-far gating.
    # The first progress-pay version paid only new episode-max progress —
    # which killed parking but also killed the SHEPHERD: after one crumble,
    # re-approaching the pose paid nothing, and a from-scratch brain never
    # ignited (513090-s1: hold flat ~0.05 even on xml servos, while the old
    # annuity recipe's scratch run hit 1.3@3M — the annuities were doing
    # real work as the come-back-this-way gradient). Symmetric deltas keep
    # both guarantees: climbing pays +, falling pays −, so parking nets 0,
    # cycling nets 0, and every re-approach still has dense gradient. An
    # episode's slope income telescopes to phi(end) − phi(start): bounded,
    # unfarmable. Baselines anchor to the first post-spawn state, so
    # handout spawns still bank nothing.
    # Persistence-ramp bookkeeping (see _headstand_hold): one increment per
    # step, here in the state hook so multiple term evaluations in a step
    # (previews, probes, telemetry) can't double-count the streak.
    raw = _headstand_hold_raw(env)
    env._hs_streak = (getattr(env, "_hs_streak", 0) + 1) if raw > 0.05 else 0
    prev = getattr(env, "_hs_prev", None)
    cur = {k: fn(env) for k, fn in _HS_POTENTIALS}
    if prev is None:
        env._hs_gain = {k: 0.0 for k in cur}
    else:
        env._hs_gain = {k: cur[k] - prev[k] for k in cur}
    env._hs_prev = cur


def _hs_gain_term(key):
    def term(env) -> float:
        gains = getattr(env, "_hs_gain", None)
        return 0.0 if gains is None else gains[key]
    term.__name__ = f"_hs_gain_{key}"
    return term


_register(Behavior(
    id="headstand",
    emoji="🙃",
    title="Do a headstand",
    description=(
        "Tip forward, plant the head on the floor, and balance upside down "
        "with the feet in the air."
    ),
    how_it_learns=(
        "The salary is the real headstand — crown on the floor, body above "
        "it, feet on top, chin tucked. Once it's over, thrashing and "
        "jerky servos are charged so it has to CATCH on the dome rather "
        "than roll past. Going onto the back ends the attempt. Resting "
        "on the face is a penalty."
    ),
    keywords=("headstand", "head stand", "handstand", "upside down",
              "invert", "on its head"),
    # Hold + tallness salaries, two shepherds, two prohibitions. The
    # orientation shepherd cannot see the inverted crumple (gz stays high
    # while the legs fold) — body_lifted / legs_straight are the per-step
    # reason to KEEP the stack. Auxiliary polish stays in the catalog.
    terms=(
        RewardTerm("headstand_hold",
                   "The one salary: head down, body above it, feet on top, all airborne",
                   8.0, _headstand_hold),
        # Per-step tallness: orientation-only shaping could not see the
        # inverted crumple (gz stays high while the legs fold). This pays
        # the stacked drop every step and drops to 0 the moment a foot
        # hits — the missing reason to KEEP balancing. Champion lineage
        # learned straight legs under this salary; progress-pay converted
        # it into a one-shot and stage-1 brains flopped (feet_up 0.13).
        RewardTerm("body_lifted",
                   "Points for keeping the body stacked tall on the head",
                   3.0, _body_lifted),
        RewardTerm("legs_straight",
                   "Points for keeping the legs straight up while upside down",
                   4.0, _legs_straight_up),
        # Champion (849641) salaries, dropped in the minimal-covenant trim.
        # Action 0 is DEFAULT_POSE (neck +0.35, the standing "head upright"
        # look). Without a per-step tuck/kick pay the student un-tucks on
        # the first step, pivots on the face, rolls onto its back, and has
        # no reason to use the neck to get the crown under and kick back
        # over — which is what 849641 kept doing (neck_tuck ~1.0/step,
        # feet_up ~1.05/step, hold 2.67). 840b8a had the same term NAMES
        # but as orientation annuities and never tucked (neck_tuck 0.01).
        # Stay-tucked salary, as 849641 (1.0/step at the hold). Progress-only
        # let it arrive tucked and then un-tuck. Face-plant / front-side
        # gates stop the old face-lean camp from collecting this.
        RewardTerm("neck_tuck",
                   "Points for keeping the chin tucked so it stays on the crown",
                   1.5, _neck_tuck),
        RewardTerm("feet_up",
                   "Points for kicking BOTH feet up overhead while upside down",
                   # 6.0, not 2.0: the weight the successful from-scratch
                   # ladder chain (eac0f9) actually trained under — it rode
                   # in as a sticky override, and shipping the proven config
                   # beats shipping an untested one. The strong unfold pull
                   # matters on the ladder's later rungs, where extension
                   # must survive the servo step-down.
                   6.0, _feet_up),
        RewardTerm("upside_down", "Points for getting MORE upside down than before", 40.0,
                   _hs_gain_term("upside_down")),
        #   2b. feet_rise — the second shepherd, for the last rung: pays
        #       each step the LOWER foot gets higher than it has been this
        #       episode (both feet must rise). Covers the prep-to-stack
        #       leap that the inversion shepherd cannot see.
        RewardTerm("feet_rise", "Points for lifting the feet higher than before", 30.0,
                   _hs_gain_term("feet_rise")),
        RewardTerm("wrong_way", "Penalty for going over backwards instead", 1.0,
                   _wrong_way_pen, is_penalty=True),
        #   4. body_drag — the one rent; a small per-step charge while the
        #      trunk/hips rest on the floor, so the measured dive-and-freeze
        #      trap costs something instead of being a free bed (paired with
        #      the 0.5 hold gate: the way OUT of the trap now also pays).
        RewardTerm("body_drag", "Small rent for lying on the ground instead of trying", 0.5,
                   _body_drag_pen, is_penalty=True),
        RewardTerm("face_plant", "Penalty for resting on the face or underside instead of the crown",
                   1.0, _face_on_floor_pen, is_penalty=True),
        # smooth_moves / gentle_joints / save_energy stay OUT of scratch
        # (same lesson as calm_up_top): on 4d93a6 they charged −1.9/step
        # against hold +1.3, so correction thrash cost more than the catch
        # paid and training stalled. Champion ran them only after the hold
        # already existed (hold 2.67 > smoothness 0.9). Polish fine-tune.
    ),
    default_steps=20_000_000,
    success_metric="time spent inverted on the head with feet airborne",
    episode_s=8.0,  # short reps: measured 8s ignites, 20s does not
    scene="all",
    terminate_on_fall=False,  # inverted IS a fall to the walk env; overshoot-onto-back is a separate terminal in BehaviorEnv.step
    state_fn=_hs_update,  # progress-pay bookkeeping for the slope terms
    # Fallback spawn mix for an UNSTAGED run of this recipe (a fine-tune,
    # or a preview env built without stage knobs). The ladder below
    # overrides these per stage — see THE LADDER IS BACK.
    inverted_spawn_prob=0.35,
    mid_flip_spawn_prob=0.30,
    # THE LADDER IS BACK (2026-09-01, the maintainer's call after the one-move
    # experiment ran its course). The single-run covenant proved a lot —
    # every income leak it forced us to find is closed and stays closed —
    # but its scratch graduates all converged to the same inverted TUCK-BALL
    # (gap ~0.015, feet ~0.14) across five configs: economics, 1.6x servos,
    # tripled unfold gradient, terminal removal, still drops. The blocker is
    # EXPLORATION, not reward: capped-noise PPO never samples a successful
    # EXTENDED catch under BAM, so its value is never learned (re-measured
    # tonight; first measured as ad85a8-s1, "BAM never ignites"). The
    # champion lineage crossed exactly this gap with the servo ladder below.
    # What is DIFFERENT from the failed staged era: the terms are identical
    # in every stage (the sealed set above — stages carry NO reward edits,
    # so no rung can grow its own leak), the shape is constitutional
    # everywhere (only DEPTH and PHYSICS ladder), and the trainer's log_std
    # cap holds across every warm-started rung (the ratchet that silently
    # poisoned yesterday's chains). Stage lessons preserved: 8 s reps, 80%
    # drops in drills, ≤15% standing in the finisher (standing-heavy finals
    # measurably destroyed working policies — see headstand-solved memory).
    curriculum=(
        CurriculumStage("holding the headstand (training wheels)", 6_000_000,
                        {"MICRODUCK_ACTUATOR": "xml",
                         "MICRODUCK_INVERTED_SPAWN_PROB": "0.80",
                         "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.10",
                         "MICRODUCK_EPISODE_S": "8",
                         "MICRODUCK_INV_SPAWN_KICK": "0.05"},
                        detail=(
                            "Almost every rep starts dropped into the stack, "
                            "nearly still, on strong phantom servos — the one "
                            "regime where a fresh brain can actually SAMPLE a "
                            "held extended headstand and learn its value.")),
        CurriculumStage("holding it on real servos", 4_000_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_BAM_CURRENT_SCALE": "1.3",
                         "MICRODUCK_INVERTED_SPAWN_PROB": "0.80",
                         "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.10",
                         "MICRODUCK_EPISODE_S": "8",
                         "MICRODUCK_INV_SPAWN_KICK": "0.10"},
                        detail=(
                            "Same drill, servos step down toward the real "
                            "XL330s (BAM model at 1.3x current) and the "
                            "drops arrive with a real shove.")),
        CurriculumStage("standing taller", 3_000_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_INVERTED_SPAWN_PROB": "0.80",
                         "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.10",
                         "MICRODUCK_EPISODE_S": "8",
                         "MICRODUCK_HS_GATE": "0.7",
                         "MICRODUCK_INV_SPAWN_KICK": "0.15"},
                        detail=(
                            "Honest servos, judged harder on DEPTH: the "
                            "closer to crown-vertical, the more each second "
                            "pays. Shape was never negotiable; now depth "
                            "tightens too.")),
        CurriculumStage("impressing the judges", 2_500_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_INVERTED_SPAWN_PROB": "0.80",
                         "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.10",
                         "MICRODUCK_EPISODE_S": "8",
                         "MICRODUCK_HS_GATE": "0.8",
                         "MICRODUCK_INV_SPAWN_KICK": "0.15"},
                        detail=(
                            "Full competition pricing: only the tall, "
                            "stacked, crown-vertical hold earns real money "
                            "(849641-s1 measured this stage straightening a "
                            "wobbly hold, 1.29 -> 2.67).")),
        CurriculumStage("finishing the kick-over", 4_000_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_INVERTED_SPAWN_PROB": "0.45",
                         "MICRODUCK_MID_FLIP_SPAWN_PROB": "0.40",
                         "MICRODUCK_EPISODE_S": "8",
                         "MICRODUCK_HS_GATE": "0.8",
                         "MICRODUCK_INV_SPAWN_KICK": "0.15"},
                        detail=(
                            "The entry: mostly rolling mid-flip catches plus "
                            "a 15% share of true standing starts — the "
                            "measured ceiling before fumbled standing reps "
                            "starve a working hold.")),
    ),
))




# ------------------------------------------------------------------ headspin
# Experimental extension of the measured headstand, not a solved skill.
# Keep the same hold/entry rewards: only spawns get easier across the ladder.

def _headspin_world_rate(env) -> float:
    """World-up angular speed from two observable body-frame vectors.

    gyro_z alone reverses sign upside down. -gravity is world UP expressed
    in the same frame as the gyro (obs[0:3], obs[3:6]).
    """
    return float(np.dot(env._gyro, -env._projected_gravity()))


def _headspin_supported(env) -> bool:
    c = env.foot_contact_state
    return (not c['left'] and not c['right']
            and _head_on_floor(env) and _body_floor_contacts(env) == 0
            and float(env._projected_gravity()[2]) > 0.8
            and _headstand_hold_raw(env) > 0.05)


def _headspin_turn(env) -> float:
    """Signed, bounded rotation income; stillness pays zero, reverse costs.

    Limit progress pay at the target so violent spins earn no extra. Count
    only a supported stack, never standing spins, face slides or somersaults.
    """
    if not _headspin_supported(env):
        return 0.0
    return float(np.clip(_headspin_world_rate(env) / 0.6, -1.0, 1.0))


def _headspin_visible(env) -> float:
    """Bounded speed-band reward: visibly turn, rather than merely balance.

    Baseline-subtracted Gaussian: zero at rest (and reverse), maximum at
    1.2 rad/s, decaying at excessive speeds. Uses only current observable
    angular velocity and support; no hidden angle/clock is rewarded.
    """
    if not _headspin_supported(env):
        return 0.0
    rate = _headspin_world_rate(env)
    baseline = float(np.exp(-((1.2 / 0.8) ** 2)))
    match = float(np.exp(-(((rate - 1.2) / 0.8) ** 2)))
    return max(0.0, (match - baseline) / (1.0 - baseline))


def _headspin_overspeed(env) -> float:
    if not _headspin_supported(env):
        return 0.0
    # The visible-speed recipe needs headroom around its 1.2 rad/s target.
    # Setting the new slider to zero retains the original 0.9 knee.
    visible_weight = env.weight_overrides.get('headspin_visible', 6.0)
    knee = 1.8 if visible_weight > 0 else 0.9
    excess = max(0.0, abs(_headspin_world_rate(env)) - knee)
    return -min(1.0, excess * excess)


def _headspin_reset(env) -> None:
    env._headspin_net = 0.0
    env._headspin_bout = 0.0
    env._headspin_best = 0.0
    env._headspin_supported_s = 0.0


def _headspin_update(env) -> None:
    _hs_update(env)
    if _headspin_supported(env):
        delta = _headspin_world_rate(env) * C.CTRL_DT
        env._headspin_net += delta
        env._headspin_bout += delta
        env._headspin_supported_s += C.CTRL_DT
        env._headspin_best = max(env._headspin_best, env._headspin_bout)
    else:
        env._headspin_bout = 0.0


def _headspin_caption(env) -> str:
    return (f"headspin {np.degrees(getattr(env, '_headspin_net', 0)):+.0f}deg "
            f"{_headspin_world_rate(env):+.2f}rad/s")


def _headspin_report(env) -> list[str]:
    return [
        f"headspin: supported net {np.degrees(env._headspin_net):+.1f} deg; "
        f"best uninterrupted directed rotation {np.degrees(env._headspin_best):.1f} deg; "
        f"head-only stacked support {env._headspin_supported_s:.2f} s",
        "Experimental proxy: jaw_soft is one body; visually verify crown contact. "
        "A spawned inverted start does not demonstrate entry from standing.",
    ]


_register(replace(
    BEHAVIORS['headstand'],
    id='headspin', emoji='🌀', title='Do a headspin',
    description='Enter a headstand, then rotate slowly on the crown with both feet off the floor.',
    how_it_learns=(
        'Starts from a completed headstand model. Keep the headstand while turning '
        'around world-up, with visible-spin pay targeting 1.2 rad/s (about 5 s/turn). Only head-supported inverted turns earn '
        'rotation points; reverse turns subtract points and excess speed costs. '
        'This experimental recipe trains continuous rotation, not a stop-and-stand finish.'),
    keywords=('headspin', 'head spin', 'spin on its head', 'spin on your head',
              '头顶旋转', '头顶接地旋转', '头转'),
    terms=BEHAVIORS['headstand'].terms + (
        RewardTerm('headspin_turn', 'Points for slow directed rotation while balanced on the head',
                   3.0, _headspin_turn),
        RewardTerm('headspin_visible', 'Visible headspin: turn at about 1.2 rad/s (one turn in 5 seconds)',
                   6.0, _headspin_visible),
        RewardTerm('headspin_overspeed', 'Penalty for spinning too fast on the head',
                   1.0, _headspin_overspeed, is_penalty=True),
    ),
    # Fixed world-up direction; retain the donor's zero command inputs.
    # This is intentionally not mirror-symmetric.
    symmetric=False,
    warm_start_behavior='headstand',
    default_steps=3_000_000, episode_s=12.0,
    success_metric='uninterrupted signed head-supported rotation, evaluated from standing',
    state_fn=_headspin_update, reset_fn=_headspin_reset, obs_fn=None,
    caption_fn=_headspin_caption, report_fn=_headspin_report,
    inverted_spawn_prob=0.45, mid_flip_spawn_prob=0.40,
    curriculum=tuple(
        CurriculumStage(label, 1_000_000, {
            'MICRODUCK_ACTUATOR': 'bam',
            'MICRODUCK_INVERTED_SPAWN_PROB': inv,
            'MICRODUCK_MID_FLIP_SPAWN_PROB': mid,
            'MICRODUCK_INV_SPAWN_KICK': kick,
            'MICRODUCK_HS_GATE': '0.8',
            'MICRODUCK_EPISODE_S': '12',
        }, detail=detail)
        for label, inv, mid, kick, detail in (
            ('slow turns from a balanced headstand', '0.85', '0.10', '0.05',
             '85% inverted starts, 10% entry practice, 5% standing. Retain balance while discovering slow rotation.'),
            ('keeping the turn through disturbances', '0.70', '0.20', '0.15',
             '70% inverted, 20% mid-entry, 10% standing; larger initial disturbances, unchanged motor strength.'),
            ('entering the headspin from standing', '0.45', '0.40', '0.15',
             '45% inverted, 40% mid-entry, 15% standing. Evaluate the finished policy separately using standing-only starts.'),
        )
    ),
))


# ------------------------------------------------------- foot-driven headspin
# A separate experiment.  Curriculum-only inverted starts may receive a yaw
# kick so the policy can first learn to KEEP a head-supported turn.  The final
# stage and deterministic evaluation remain standing-only with zero kick.
_LAUNCH_TURNS = 4.0
_LAUNCH_SUCCESS_TURNS = 1.0
_LAUNCH_RATE = 5.0                 # rad/s; fast-spin target, not measured ability
_LAUNCH_LZ = 0.015                # kg m^2/s; launch shaping reference


def _launch_lz(env):
    # Recompute rather than reading potentially stale subtree velocity caches.
    mujoco.mj_subtreeVel(env.model, env.data)
    return float(env.data.subtree_angmom[env.trunk_body_id, 2])


def _launch_foot_torque(env):
    """World-z ground moment about whole-body COM, FOOT contacts only.

    Sampled at control frequency; an impulse proxy, not substep-exact accounting.
    Contact frame rows are world axes; mj_contactForce acts on geom2.
    """
    torque = 0.0
    com = env.data.subtree_com[env.trunk_body_id]
    feet = set(env.foot_geoms.values())
    force = np.zeros(6)
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 == env.floor_geom and g2 in feet:
            sign = 1.0
        elif g2 == env.floor_geom and g1 in feet:
            sign = -1.0
        else:
            continue
        mujoco.mj_contactForce(env.model, env.data, i, force)
        frame = contact.frame.reshape(3, 3)
        world_force = sign * (frame.T @ force[:3])
        world_torque = sign * (frame.T @ force[3:])
        torque += float(np.cross(contact.pos - com, world_force)[2] + world_torque[2])
    return torque


def _launch_reset(env):
    env._ls_target_turns = float(np.clip(
        float(_spawn_knob(env, "MICRODUCK_LAUNCH_TURNS", str(_LAUNCH_TURNS))), .125, 5.0))
    env._ls_support_gap_max = float(np.clip(
        float(_spawn_knob(env, "MICRODUCK_LAUNCH_SUPPORT_GAP", "0.6")), .1, 1.5))
    yaw_kick = max(0.0, float(_spawn_knob(env, "MICRODUCK_LAUNCH_YAW_KICK", "0")))
    assisted = env.last_spawn != 'standing' and yaw_kick > 0
    if env.last_spawn == 'standing':
        # The real launch test always begins motionless.  Reverse-curriculum
        # spawns keep their entry velocity and can add a declared yaw kick.
        env.data.qvel[:] = 0.0
    elif assisted:
        env.data.qvel[5] += yaw_kick * env._rng.uniform(.85, 1.0)
    mujoco.mj_forward(env.model, env.data)
    env._ls_phase = 1 if env.last_spawn == 'inverted' and assisted else 0
    env._ls_time = env._ls_turn = env._ls_head_turn = 0.0
    env._ls_quality = env._ls_takeoff_lz = env._ls_peak_lz = 0.0
    env._ls_peak_rate = 0.0
    env._ls_launch_gain = env._ls_spin_gain = 0.0
    env._ls_gap = env._ls_settled = 0.0
    env._ls_foot_impulse = 0.0
    env._ls_prev_lz = _launch_lz(env)
    env._ls_prev_feet = False
    env._ls_foot_seen = assisted
    # Assisted starts explicitly expose their training wheel in the command
    # observation.  Standing starts must still earn quality from foot impulse.
    env._ls_quality = 1.0 if assisted else 0.0
    env._ls_success = False
    env._ls_head_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, 'jaw_soft')
    env._ls_head_velocity = np.zeros(6)


def _launch_update(env):
    _hs_update(env)
    env._ls_time += C.CTRL_DT
    feet = any(env.foot_contact_state.values())
    lz = _launch_lz(env)
    env._ls_launch_gain = env._ls_spin_gain = 0.0
    if feet:
        env._ls_foot_seen = True
    if env._ls_phase == 0:
        # Only new angular-momentum records during FOOT contact earn pay.
        # Bounded total launch pay; repeatedly winding/unwinding cannot farm it.
        if feet and env._ls_time <= 3.0:
            env._ls_foot_impulse += _launch_foot_torque(env) * C.CTRL_DT
            clipped = float(np.clip(min(lz, env._ls_foot_impulse), 0, _LAUNCH_LZ))
            env._ls_launch_gain = max(0, clipped - env._ls_peak_lz) / _LAUNCH_LZ
            env._ls_peak_lz = max(env._ls_peak_lz, clipped)
        if env._ls_prev_feet and not feet:
            # Use the LAST foot-contact value, not post-takeoff neck/leg motion.
            env._ls_takeoff_lz = env._ls_prev_lz
            env._ls_quality = float(np.clip(min(env._ls_takeoff_lz, env._ls_foot_impulse) / _LAUNCH_LZ, 0, 1))
            if env._ls_foot_impulse < 0.6 * max(0, env._ls_takeoff_lz):
                env._ls_quality = 0.0
        if _headspin_supported(env) and env._ls_foot_seen:
            env._ls_phase = 1
        elif env._ls_time > 4.0:
            env._ls_phase = 2
    elif env._ls_phase == 1:
        if _headspin_supported(env):
            env._ls_gap = 0.0
            old = min(env._ls_turn, env._ls_head_turn)
            rate = _headspin_world_rate(env)
            env._ls_peak_rate = max(env._ls_peak_rate, rate)
            env._ls_turn += rate * C.CTRL_DT
            mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY,
                                    env._ls_head_id, env._ls_head_velocity, 0)
            env._ls_head_turn += float(env._ls_head_velocity[2]) * C.CTRL_DT
            # Both HEAD and TRUNK must rotate: neck-only twisting cannot finish.
            new = min(env._ls_turn, env._ls_head_turn)
            env._ls_spin_gain = float(np.clip((new - old) / (_LAUNCH_RATE * C.CTRL_DT), -1, 1))
            if new >= _LAUNCH_SUCCESS_TURNS * 2 * np.pi:
                env._ls_success = True
            if new >= env._ls_target_turns * 2 * np.pi:
                env._ls_phase = 2
        else:
            env._ls_gap += C.CTRL_DT
            if env._ls_gap > env._ls_support_gap_max:
                env._ls_phase = 2   # no sum of disconnected half-turn attempts
        if env._ls_time > 10.0:
            env._ls_phase = 2
    if env._ls_phase == 2:
        # Recovery pose is deliberately unconstrained.  One legitimate,
        # continuous turn is success; reaching two to four turns earns more
        # dense rotation reward before this phase begins.
        env._ls_settled = 0.0
    env._ls_prev_lz, env._ls_prev_feet = lz, feet


def _launch_obs(env):
    if not hasattr(env, '_ls_phase'):
        return
    # Six task commands (55:61), not a changed observation/action layout.
    # Contact/whole-body momentum are simulator-derived task-controller signals;
    # hardware would need estimators. This is a local simulation experiment.
    progress = min(env._ls_turn, env._ls_head_turn) / (env._ls_target_turns * 2 * np.pi)
    env.body_cmd[:] = [env._ls_phase / 2, np.clip(1 - progress, -1, 1),
                       min(env._ls_time / 12, 1), env._ls_quality,
                       np.clip(_launch_lz(env) / _LAUNCH_LZ, -2, 2),
                       float(any(env.foot_contact_state.values()))]


def _launch_entry_term(fn):
    def gated(env):
        # Keep the donor's balance skill alive through the spin.  It switches
        # off only for recovery, where indefinite headstanding must not win.
        return fn(env) if env._ls_phase != 2 else 0.0
    return gated


def _launch_push(env):
    return env._ls_launch_gain


def _launch_spin(env):
    # Later turns are worth progressively more, so the easy one-turn solution
    # remains a success but is not the reward optimum.  The multiplier reaches
    # 2x during the fourth turn and never pays without new positive rotation.
    turns = max(0.0, min(env._ls_turn, env._ls_head_turn) / (2 * np.pi))
    multiplier = 1.0 + min(turns, 3.0) / 3.0
    return env._ls_quality * env._ls_spin_gain * multiplier


def _launch_balance(env):
    if env._ls_phase != 1:
        return 0.0
    return _headstand_hold_raw(env) * env._ls_quality


def _launch_overspeed(env):
    return -min(1, max(0, abs(_headspin_world_rate(env)) - 8.0) ** 2)


def _launch_caption(env):
    phase = ('push', 'spin', 'recover')[getattr(env, '_ls_phase', 0)]
    turns = min(getattr(env, '_ls_turn', 0), getattr(env, '_ls_head_turn', 0)) / (2 * np.pi)
    target = getattr(env, '_ls_target_turns', _LAUNCH_TURNS)
    return (f'{phase}: {turns:.2f}/{target:g} turns; '
            f'peak {getattr(env, "_ls_peak_rate", 0):.2f} rad/s; '
            f'foot launch {getattr(env, "_ls_takeoff_lz", 0):.4f} kg m2/s')


def _launch_report(env):
    return [_launch_caption(env), f'proxy_success_1plus_turn={env._ls_success}',
            'No externally injected spin. Head/trunk world-up gyro integrals are proxies; verify video.']


_register(replace(
    BEHAVIORS['headstand'], id='headspin_launch', emoji='🌀', title='Foot-driven headspin',
    description='Push and twist from the feet, then carry a fast headspin for one to four turns.',
    how_it_learns='Warm-starts from the trained headspin. One continuous foot-launched turn is success; faster second through fourth turns earn more. The reverse curriculum removes the yaw assist and finishes from a motionless standing start. Experimental: success has not been demonstrated.',
    keywords=('headspin_launch', 'foot-driven headspin', '撑地启动头转', '撑地头转'),
    terms=tuple(replace(t, fn=_launch_entry_term(t.fn)) for t in BEHAVIORS['headstand'].terms) + (
        RewardTerm('launch_push', 'Build initial rotation while the feet still push on the floor', 24.0, _launch_push),
        RewardTerm('launch_spin', 'Carry fast launch momentum through one to four head-supported turns; later turns pay more', 32.0, _launch_spin),
        RewardTerm('launch_balance', 'Keep the headstand while rotating after the launch', 8.0, _launch_balance),
        RewardTerm('launch_overspeed', 'Penalty for excessive rotation above 8 rad/s', 2.0, _launch_overspeed, is_penalty=True),
    ),
    warm_start_behavior='headspin', symmetric=False, default_steps=3_000_000,
    episode_s=10.0, inverted_spawn_prob=0.0, mid_flip_spawn_prob=0.0,
    curriculum=tuple(
        CurriculumStage(label, 750_000, {
            'MICRODUCK_ACTUATOR': 'bam', 'MICRODUCK_BAM_CURRENT_SCALE': '1.0',
            'MICRODUCK_INVERTED_SPAWN_PROB': inv,
            'MICRODUCK_MID_FLIP_SPAWN_PROB': mid,
            'MICRODUCK_INV_SPAWN_KICK': entry_kick,
            'MICRODUCK_LAUNCH_YAW_KICK': yaw_kick,
            'MICRODUCK_LAUNCH_TURNS': turns,
            'MICRODUCK_LAUNCH_SUPPORT_GAP': '0.6',
            'MICRODUCK_HS_GATE': '0.8', 'MICRODUCK_EPISODE_S': episode,
        }, detail=detail)
        for label, inv, mid, entry_kick, yaw_kick, turns, episode, detail in (
            ('preserve a fast half-turn from head support', '0.85', '0.10', '0.05', '1.5', '0.5', '8',
             'Start from the trained headspin pose with a visible yaw kick; retain a fast half-turn.'),
            ('carry one successful turn through entry', '0.65', '0.25', '0.10', '1.0', '1.0', '9',
             'Reduce the yaw assist and learn the one-turn success threshold.'),
            ('extend launch momentum through two turns', '0.30', '0.30', '0.10', '0.5', '2.0', '10',
             'Standing starts become the largest group while later turns earn progressively more.'),
            ('motionless foot launch for four turns', '0', '0', '0', '0', '4.0', '12',
             'Final stage starts motionless on the feet with no injected yaw; one turn succeeds and two to four are preferred.'),
        )
    ),
    spawn_families=(), spotter_fn=None,
    state_fn=_launch_update, reset_fn=_launch_reset, obs_fn=_launch_obs,
    caption_fn=_launch_caption, report_fn=_launch_report,
    success_metric='standing start and at least one continuous head AND trunk turn; foot-launch quality is reported separately, and two to four fast turns are preferred',
))


# Star-export EVERYTHING (helpers included) so downstream modules and the
# package __init__ can reassemble the old flat-module surface exactly.
__all__ = [n for n in dir() if not n.startswith("__")]
