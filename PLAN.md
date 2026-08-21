# PLAN.md — NeuRelux

**Physics-Guided, Physics-Informed & Physics-Encoded Neural Networks for the ATLAS Magnetic Track Brake**

_Last updated 2026-08-20._ This is the working plan for the project — the reasoning behind the structure, the method definitions, and the notebook build order. Update it as scope changes, datasets turn out to be unavailable, or architectural decisions get made.

---

## 0. Where this starts from

No ATLAS documentation, MATLAB/Simulink files, FEM exports, or measurement data exist anywhere in this workspace yet — just the new `neurelux/` project itself alongside a set of unrelated robotics/racing repos.

Consequences for this plan:

- Sections 1 ("Existing ATLAS model") and 2 ("Available ATLAS measurements") below are **placeholders based on standard textbook equivalent-magnetic-circuit practice** for magnetic track brakes (as used e.g. in railway eddy-current / magnetic-track-brake literature), not on an inspected ATLAS document. Assumed content is marked accordingly.
- If ATLAS documentation, MATLAB/Simulink files, FEM exports, or measurement CSVs become available, drop them into `data/raw/atlas/` and this section gets rewritten from the real source. Until then, all notebooks use synthetic data and public benchmarks only, and no ATLAS experimental value is invented — so the missing documentation doesn't block the notebook plan below.

---

## 1. Existing ATLAS model

*Assumed — based on general magnetic-track-brake practice, not inspected ATLAS documentation; see §0.*

A magnetic track brake (Elektromagnetische Schienenbremse) is a non-contact-actuated friction brake: DC-excited electromagnets are lowered onto the rail, generate an attractive normal force, and the resulting friction (magnet skid sliding along the rail head) provides the braking force. Eddy currents induced by relative motion between magnet and rail both alter the field distribution (motion-induced skin effect) and contribute their own retarding force component. The assumed ATLAS model follows the standard reduced-order approach used for real-time / control-oriented simulation of such systems:

- **Equivalent magnetic circuit (lumped reluctance network).** MMF source `Θ = N·I` drives flux `Φ` through a loop of series/parallel reluctances: pole reluctance, air-gap reluctance `R_gap = g/(μ₀A)`, yoke/rail steel reluctance `R_Fe(B)` with a piecewise-linear or lookup-table saturation curve, and leakage reluctance. This is a 0D/lumped model — no explicit spatial field distribution.
- **Layered equivalent network for skin/eddy-current effects.** Because a solid steel rail head does not behave as a single lumped element under time-varying or motion-induced excitation, the rail (and possibly the pole faces) is subdivided into `N` layers through the depth, each represented by a resistance–reluctance pair, forming a ladder (Cauer-type) network. This reproduces frequency- and velocity-dependent flux penetration depth without solving a full 2D/3D eddy-current PDE.
- **Force equation.** Magnetic attraction force approximated from co-energy / Maxwell stress at the gap, typically `F_A ≈ Φ²/(2μ₀A_eff)` per pole face, summed over poles.
- **Friction model.** Coulomb-type: `F_R = μ·F_N` per pole, `F_R,total = Σᵢ μᵢ·F_A,i`, with `μ` a function of relative velocity (and possibly temperature, wear) from a lookup table or empirical fit — not from first principles.
- **Temperature model.** Lumped thermal capacitance fed by resistive (I²R) and friction (`|F_R·v|`) losses, dissipated to ambient via a convective term; temperature feeds back into steel/rail conductivity and into the friction coefficient.
- **Known limitations (assumed, typical of this model class):** linear/piecewise saturation is a coarse fit outside its calibration range; the layered skin-effect network has a fixed layer count/thickness chosen a priori rather than learned from data; motion-induced eddy currents (the `v×B` term) are typically folded into empirical velocity-dependent correction factors rather than derived from the governing PDE; friction coefficient vs. velocity/temperature is empirical and a major source of validation error; no explicit handling of multi-pole magnetic coupling.

**This section must be rewritten from the real ATLAS document as soon as it is available.**

## 2. Available ATLAS measurements

*Assumed — see §0.*

Typical instrumentation for this class of system, listed here as the expected signal set that `interfaces/simulink/model_io.yaml` and Notebook 09's ATLAS data loader should be able to consume once real data arrives:

| Signal | Symbol | Unit |
|---|---|---|
| Coil current | `I` | A |
| Coil voltage | `U` | V |
| Air gap | `g` | m |
| Rail-relative velocity | `v` | m/s |
| Magnet/rail temperature | `T` | °C / K |
| Magnetic flux (search coil or Hall) | `Φ` or `B` | Wb / T |
| Normal (attraction) force | `F_A` | N |
| Braking force | `F_R` | N |

**Open question:** confirm which of these are actually logged, at what sample rate, and over what operating envelope (current range, gap range, velocity range, temperature range) — this directly determines which extrapolation holdouts (Section 18 style, see below) are meaningful.

---

## 3. Candidate physics-guided methods

**Definition — PHYSICS-GUIDED:** physical variables, engineered features, or auxiliary physical models are used to *inform the input/output structure or training signal* of an otherwise black-box learner, but no physical equation constrains the network's internal computation and no physical equation appears in the loss.

| Method | What is guided |
|---|---|
| Feature engineering from physics (`Θ=NI`, `1/g`, `B_sat` proximity, Reynolds-type magnetic number `μ₀σvL`) fed to an MLP | Input representation |
| Physics-guided residual NN: `y = y_physics_baseline + r_θ(x)` | Output structure — network only learns the *correction* to a cheap physical baseline (e.g., the assumed ATLAS linear circuit) |
| Physics-guided train/test splits (Section 18) | Evaluation protocol, not the model itself |
| Dimensionless/normalized inputs derived from magnetic circuit theory | Preprocessing |

## 4. Candidate physics-informed methods

**Definition — PHYSICS-INFORMED (PINN-style):** the network architecture is generic (MLP, GRU, …), but a physical governing equation (PDE residual, conservation law, constitutive relation) is added as an **additional loss term**, penalizing violations at collocation points or measured samples.

| Method | Governing equation used as loss |
|---|---|
| Maxwell/vector-potential PINN | `∇×(ν∇×A) = J`, quasi-static eddy-current PDE residual |
| Magnetic diffusion PINN | `μ ∂H/∂t = ∂/∂z(ρ ∂H/∂z)` residual at collocation points in `(z,t)` |
| Energy-consistency loss | Penalize `λ - ∂W'/∂I ≠ 0` and `F_A + ∂W'/∂g ≠ 0` if flux/force are predicted by separate heads |
| Jiles–Atherton-informed loss | Penalize deviation from JA ODE `dM/dH = …` even if the primary predictor is a GRU |

## 5. Candidate physics-encoded methods

**Definition — PHYSICS-ENCODED:** the physical equation/topology is **built directly into the forward computation graph** — it cannot be violated by construction (up to numerical integration error), and only physically-meaningful sub-quantities (reluctances, capacitances, conductances, a single co-energy potential) are learned, always through a positivity-preserving parameterization.

| Method | Structure encoded |
|---|---|
| Neural equivalent magnetic circuit | `Φ = Θ/R_total(B,T)`, series/parallel reluctance topology fixed, only `R(·)` learned |
| Neural reluctance graph | Graph whose edges *are* reluctances/permeances; Kirchhoff flux/MMF laws hold exactly |
| Cauer neural network (1D) | `C dx/dt = -Dᵀ G D x + B_u u`, ladder topology fixed, `C_i,G_i>0` learned via softplus |
| Graph-Cauer (surface × depth) | 2D graph (depth ladder × surface neighbors), same conservation form, local parameters learned |
| Co-energy network | Single scalar potential `W'_θ(I,g,T)`; flux and force are *exact* partial derivatives of the same function (autograd), so they cannot become mutually inconsistent |
| Port-Hamiltonian NN | Energy + dissipation structure (`ẋ = (J-R)∇H`) encoded, guarantees passivity |

**These three categories are not mutually exclusive within one architecture** — the recommended final ATLAS model (Section 8) is guided (feature choices), informed (optional energy/PDE residual regularizers on top), *and* encoded (topology) simultaneously, at different levels.

**Two flavors of physics-encoded.** The Cauer ladder only learns scalar *parameters* (`C_i, G_i`) inside a fixed linear equation — no neural network sits inside the ODE. The neural equivalent circuit and the Jiles–Atherton-plus-residual model go one step further: an actual small neural network is embedded directly in the equation's right-hand side (`μ_θ(B,T)`, or a residual added to `dM/dH`), trained end-to-end through the integrator via autograd. This second flavor is a concrete instance of a **Universal Differential Equation** (Rackauckas et al., 2020 — verified in `papers/references.bib`): a known governing equation with an unknown term replaced or augmented by a neural network, rather than the equation being approximated by a neural network wholesale. Notebook 02's physics-encoded model (Section 7) is built this way.

---

## 6. Public datasets and benchmarks

| Dataset | URL | Data type | Variables | License | Auto-download | Notebook | ATLAS relevance |
|---|---|---|---|---|---|---|---|
| Princeton MagNet | https://github.com/PrincetonUniversity/magnet | B(t), H(t), core loss, various ferrite materials/waveforms | B, H, f, T, P_loss | research use, see repo | Partial (git-based, large) | 02 | Hysteresis modeling method testbed (ferrite, not steel — used for method validation only) |
| MagNet Challenge 2 | https://github.com/minjiechen/magnetchallenge-2 | Extended MagNet-style waveform/loss data | B, H, f, T | see repo | Partial | 02 | Same as above |
| UPB Material Database | https://github.com/upb-lea/materialdatabase | Structured magnetic material DB (permeability, loss, B-H) | material params | GPL-3.0 (confirmed) | Yes (`pip`, used) | 02 | Real N87 (Epcos) B-H major loops at 25C/100C used directly in Notebook 02 |
| FEM Magnetics Toolbox | https://github.com/upb-lea/FEM_Magnetics_Toolbox | FEM-based magnetic component design/simulation tool | geometry, field results | MIT (verify) | Yes (`pip`) | reference only | Possible future FEM cross-check |
| HystRNN | https://github.com/chandratue/HystRNN | Code + example hysteresis data/model | B, H | see repo | Yes (git clone) | 02 | Reference "physics-aware recurrent model" implementation (Model C) |
| Magnetic Hysteresis Neural Operator | https://github.com/chandratue/magnetic_hysteresis_neural_operator | Code + data for operator-learning hysteresis | B, H | see repo | Yes (git clone) | 02 (optional RIFNO/FNO) | Reference for FNO/neural-operator hysteresis baseline |
| TEAM Workshop Problem 7 | (search: "TEAM Problem 7 Testing Electromagnetic Analysis Methods") | Eddy-current benchmark, conducting plate with hole, asymmetric coil excitation | B_z at probe grid, coil geometry | benchmark, freely published | No — manual per README | 05 | Canonical eddy-current/skin-effect validation case, directly tests Graph-Cauer |
| TEAM Workshop Problem 28 | (search: "TEAM Problem 28 moving conductor") | Moving-conductor eddy-current + force benchmark | force vs. velocity, field | benchmark, freely published | No — manual per README | 06 | Closest public analogue to ATLAS's velocity-dependent eddy-current force problem |
| Kaggle: magnetic field measurement data | https://www.kaggle.com/datasets/shulmandavid/magnetic-field-measurement-data | Permanent-magnet field measurements | B(x,y,z) | Kaggle terms | Requires Kaggle API key | reference only | Illustrative sensor-data example, low direct ATLAS relevance |
| Kaggle: Electric Motor Temperature | https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature | Time series of motor temps, currents, torque, speed | T, I, n, torque | CC0 | Requires Kaggle API key | 08 | **Not an ATLAS/friction dataset** — used only to demonstrate the thermal submodel training loop |
| Electrical steel B-H / hysteresis (Mendeley/university repositories) | to be identified during Notebook 02 work | B-H loop measurements on Si-steel | B, H | varies | Case-by-case | 02 | Preferred over ferrite data — ATLAS uses steel, not ferrite |

Datasets requiring manual download or credentials must never block a notebook: each notebook checks for local data first and prints the exact `scripts/download_*.py` command (or manual instructions) if missing — see `scripts/README.md`.

---

## 7. Small notebook experiments

See Section "Notebook style" in `README.md` for the mandatory 14-part structure. One method (or a small, comparable set of methods) per notebook, synthetic-first, public-data-second:

| # | Notebook | Core question |
|---|---|---|
| 00 | `00_overview_methods.ipynb` | Which method family for which sub-problem? (survey + recommendation, no training) |
| 01 | `01_skin_effect_cauer_synthetic.ipynb` | Can a physics-encoded 1D Cauer ladder learn its own layer `C,G` from synthetic diffusion data? |
| 02 | `02_hysteresis_material_model.ipynb` | MLP vs. GRU vs. physics-aware recurrent vs. JA+residual on public (steel-preferred) B-H data |
| 03 | `03_neural_reluctance_circuit.ipynb` | Replacing only `R_Fe(B,T)` in a classical circuit vs. full black-box |
| 04 | `04_graph_cauer_surface_depth.ipynb` | Does the 2D surface×depth graph beat independent ladders / generic GNN? |
| 05 | `05_eddy_current_team7.ipynb` | Graph-Cauer / PINN against TEAM 7 public FEM/measurement reference |
| 06 | `06_moving_conductor_team28.ipynb` | Velocity-dependent eddy force against TEAM 28 |
| 07 | `07_energy_consistent_force.ipynb` | Co-energy network vs. independent flux/force heads — consistency & extrapolation |
| 08 | `08_friction_temperature_model.ipynb` | Physics-guided friction residual + lumped thermal feedback loop |
| 09 | `09_atlas_small_combined_model.ipynb` | Chain 01–08 on synthetic ATLAS-like data (real data only if/when available) |

Notebooks are built and verified one at a time, in this order, rather than all at once — each one has to actually run before the next gets started. See `README.md` for current status.

---

## 8. Final ATLAS architecture (target, not yet built)

```
            I, U, g, v, T
                  |
                  v
     Nonlinear Magnetic Circuit        <- physics-encoded (Sec. 5)
                  |
                  v
          Reluctance Graph             <- physics-encoded
                  |
                  v
   Surface x Depth Graph-Cauer         <- physics-encoded, physics-guided velocity input
                  |
                  v
            magnetic flux Phi_i
                  |
                  v
     Co-energy / attraction force      <- physics-encoded (autograd of one potential)
                  |
                  v
              F_A for each pole
                  |
                  v
        friction residual model        <- physics-guided (mu_base) + informed (positivity)
                  |
                  v
       F_R,i = mu_i * F_A,i            <- hard-coded, never learned
                  |
                  v
             sum over poles
                  |
                  v
             braking force
                  |
                  v
           thermal dynamics            <- physics-encoded lumped RC
          +-------+-------+
          |               |
          v               v
      sigma(T)        friction(T)      <- feedback into circuit & friction
```

Recommended core (from Notebook 00 evaluation, provisional): **nonlinear neural reluctance graph + graph-Cauer skin-effect model + optional hysteresis state + energy-consistent force + physics-guided friction + thermal feedback**, i.e. combination `M7`/`M8` in the ablation ladder below — but this is a hypothesis to be confirmed empirically in Notebook 00's comparison and revisited after Notebooks 01–08 produce evidence, not a foregone conclusion.

## 9. Ablation study (design, executed only after Notebooks 01–08 exist)

| ID | Model | Adds |
|---|---|---|
| M0 | Black-box MLP `(I,v,g,T)→F_A` | — |
| M1 | Original linear equivalent magnetic circuit | baseline physics, no learning |
| M2 | Nonlinear neural magnetic circuit | learned `μ(B,T)` |
| M3 | Neural reluctance graph | graph topology |
| M4 | M3 + 1D Cauer skin model | depth ladder |
| M5 | M3 + surface×depth Graph-Cauer | 2D graph |
| M6 | M5 + hysteresis material state | memory |
| M7 | M6 + co-energy-consistent force | energy consistency |
| M8 | M7 + friction + thermal feedback | full loop |
| (ref) | PINN / Maxwell surrogate | independent reference, not part of the M-ladder |

Metrics: interpolation accuracy, extrapolation accuracy (per Section 18 holdouts), physical-residual violation, parameter count, train time, inference time. Complexity is a cost to be justified by measured gains, not assumed.

## 10. Simulink integration strategy

No MATLAB dependency now. `interfaces/simulink/` defines the deployment contract every model in `src/atlas_physics/` must satisfy:

```python
state = model.initial_state()
outputs, next_state = model.step(inputs, state, dt)
```

- All internal states (Cauer ladder states, hysteresis state, thermal state) are explicit and exposed — nothing hidden inside an RNN's opaque hidden vector without a physical meaning.
- Fixed-step-compatible: `step()` must be a pure function of `(inputs, state, dt)`, no internal adaptive-step solvers.
- SI units at every interface boundary (`model_io.yaml`).
- Deterministic golden test vectors (`interfaces/simulink/test_vectors/`) generated from the PyTorch reference so a future Simulink S-function port can be checked bit-for-tolerance against it.
- Submodels stay swappable: a Simulink harness should be able to substitute the neural reluctance graph with the original linear circuit, or the Graph-Cauer block with a classical fixed Cauer ladder, without touching the surrounding benchmark scaffolding — this is why each `src/atlas_physics/*.py` module exposes the same `step()`-shaped interface regardless of whether it's "classical" or "neural".

---

## Open items / assumptions to revisit

1. Real ATLAS documentation not yet available — Sections 1–2 need rewriting once provided.
2. TEAM 7 / TEAM 28 reference data availability needs to be checked at Notebook 05/06 time (no automatic download exists for these benchmarks as far as currently known — treated as manual-download-with-instructions, see `scripts/README.md`).
3. Electrical-steel B-H dataset for Notebook 02 needs to be identified (Mendeley/university search) — ferrite (MagNet) used as fallback/method-validation only.
4. License text for UPB repos and FEM Magnetics Toolbox should be confirmed (currently assumed MIT, verify in `REFERENCES.md` before any code reuse beyond inspiration).
5. Liquid Time-constant Networks (`REFERENCES.md` "Further reading") are a candidate continuous-time architecture for Notebook 02's memory/hysteresis models, given their structural similarity to the Cauer ladder's own ODE form — worth a look once Notebook 02 is underway, not yet adopted.
