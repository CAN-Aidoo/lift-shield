"""Addendum A5: illustrative renders (MuJoCo, OSMesa). Replays recorded qpos of real episodes in a copy of the model
that adds only visual geometry (no contact, same joints), so the pictures show exactly the simulated motion."""
import os; os.environ.setdefault("MUJOCO_GL", "osmesa")
import sys, json, numpy as np, mujoco, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import lift_env as L
from lift_env import LiftRig, run_episode, LinearPolicy, sample_object

VIS = """
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/><quality shadowsize="4096"/><global offwidth="1200" offheight="1200"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.98 0.98 1" rgb2="0.78 0.82 0.88" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.93 0.93 0.95" rgb2="0.86 0.87 0.9" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="12 12" reflectance="0.05"/>
    <material name="metal" rgba="0.55 0.6 0.66 1" specular="0.6" shininess="0.6"/>
    <material name="dark" rgba="0.18 0.22 0.28 1" specular="0.3"/>
    <material name="rod" rgba="0.85 0.45 0.2 1" specular="0.4"/>
    <material name="pad" rgba="0.15 0.5 0.75 1"/>
  </asset>"""
EXTRA_PALM = """<geom type="box" size="0.045 0.018 0.012" pos="0 0 0.094" material="dark" contype="0" conaffinity="0" mass="0"/>
        <geom type="cylinder" size="0.012 0.12" pos="0 0 0.226" material="metal" contype="0" conaffinity="0" mass="0"/>
        <geom type="box" size="0.022 0.0015 0.0012" pos="0 0.0125 0.08" rgba="0.85 0.1 0.15 1" contype="0" conaffinity="0" mass="0"/>"""
EXTRA_LF = '<geom type="box" size="0.004 0.012 0.045" pos="-0.0342 0 0.045" material="metal" contype="0" conaffinity="0" mass="0"/>'
EXTRA_RF = '<geom type="box" size="0.004 0.012 0.045" pos="0.0342 0 0.045" material="metal" contype="0" conaffinity="0" mass="0"/>'


def render_model(m, mu):
    x = L.XML.format(m=m, mu=mu)
    x = x.replace("<mujoco>", "<mujoco>" + VIS, 1)
    x = x.replace("<worldbody>", '<worldbody>\n    <light pos="0.3 -0.4 1.2" dir="-0.3 0.4 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>\n'
                  '    <geom type="plane" size="1 1 0.01" pos="0 0 0.3" material="floor" contype="0" conaffinity="0"/>\n'
                  '    <camera name="cam" mode="fixed" pos="0.16 -0.5 0.6" xyaxes="0.952 0.305 0 -0.02 0.06 0.998" fovy="40"/>', 1)
    x = x.replace('contype="0" conaffinity="0"/>\n      <body name="lf">', 'contype="0" conaffinity="0"/>\n        ' + EXTRA_PALM + '\n      <body name="lf">', 1)
    x = x.replace('solimp="0.95 0.99 0.0005"/></body>\n      <body name="rf">', 'solimp="0.95 0.99 0.0005"/>' + EXTRA_LF + '</body>\n      <body name="rf">', 1)
    x = x.replace('name="rpad"', 'name="rpad" material="pad"').replace('name="lpad"', 'name="lpad" material="pad"')
    x = x.replace('solimp="0.95 0.99 0.0005"/></body>\n    </body>', 'solimp="0.95 0.99 0.0005"/>' + EXTRA_RF + '</body>\n    </body>', 1)
    x = x.replace('name="rodg" type="box"', 'name="rodg" material="rod" type="box"')
    return mujoco.MjModel.from_xml_string(x)


QP = None
_mj1 = LiftRig.mj1
def rec(self):
    _mj1(self)
    if QP is not None: QP.append(self.data.qpos.copy())
LiftRig.mj1 = rec


def episode(obj, r, mode, wname, seed, i):
    global QP
    QP = []
    pol = LinearPolicy(np.load(f"weights/w_{wname}_s{seed}.npy")) if wname else None
    res = run_episode(dict(obj, ratio=r), np.random.default_rng(5000 + i), mode, pol)
    q = np.array(QP); QP = None
    return res, q


def frames(model, q, idx, w=360, h=480):
    data = mujoco.MjData(model); rd = mujoco.Renderer(model, h, w); out = []
    for k in idx:
        data.qpos[:] = q[k]; mujoco.mj_forward(model, data); rd.update_scene(data, camera="cam"); out.append(rd.render().copy())
    return out


if __name__ == "__main__":
    tr = json.load(open("results/extended/traces.json")); i = tr["objects"][0]
    rng = np.random.default_rng(11); objs = [sample_object(rng) for _ in range(40)]; o = objs[i]
    model = render_model(o["m"], o["mu"])
    rows = []
    for lab, mode, wname, seed in (("B2 residual (no shield)", "residual", "residual", 2), ("B5 shielded residual", "residualS", "residualS", 2)):
        res, q = episode(o, 1.2, mode, wname, seed, i)
        n = len(q); nl = int(round(res["T"] / L.DT)); start = n - nl - 300
        idx = [start - 1, start + nl // 3, start + 2 * nl // 3, start + nl - 1, n - 1]
        rows.append((lab, res, frames(model, q, idx)))
        hi = frames(model, q, [idx[0], idx[-1]], w=600, h=800)
        plt.imsave(f"paper/hero_{mode}_start.png", hi[0]); plt.imsave(f"paper/hero_{mode}_end.png", hi[1])
    fig, ax = plt.subplots(2, 5, figsize=(7.2, 3.9))
    titles = ["before lift", "1/3 of lift", "2/3 of lift", "end of lift", "after 0.3 s hold"]
    for r_, (lab, res, fr) in enumerate(rows):
        for c in range(5):
            ax[r_, c].imshow(fr[c]); ax[r_, c].set_xticks([]); ax[r_, c].set_yticks([])
            for sp in ax[r_, c].spines.values(): sp.set_visible(False)
            if r_ == 0: ax[r_, c].set_title(titles[c], fontsize=7)
        ax[r_, 0].set_ylabel(f"{lab}\nslip {1000*res['slip']:.1f} mm", fontsize=7)
    plt.tight_layout(pad=0.2); plt.savefig("paper/fig_render.pdf", dpi=220); plt.savefig("paper/fig_render.png", dpi=150)
    json.dump({"object": i, "rows": [(lab, float(res["slip"]), float(res["T"]), float(res["F_rel"])) for lab, res, _ in rows]},
              open("results/extended/render.json", "w"), indent=1)
    print("render done", [(l, round(1000 * r["slip"], 2)) for l, r, _ in rows])
