# Production Management under Ecological Concern — Code Repository

This repository contains the implementation accompanying the paper:

> **Production management under ecological concern**  
> Christophette Blanchet-Scalliet, Diana Dorobantu, Caroline Hillairet, Ying Jiao, Wajdi Maatouk  

The code implements deep learning–based solvers for Hamilton–Jacobi–Bellman (HJB) equations arising in continuous-time stochastic control models of production under environmental regulation. The models are solved using neural-network approximations of the value function and the optimal control, following the **Deep Galerkin Method (DGM)** and its **Policy-Iteration-Auxiliary (PIA)** extension.

---

## Repository Layout

```
├── eco_hjb_cstC.py        
├── eco_hjb_cirC.py       
├── eco_hjb_cirC_reduced.py
├── DGM.py                
├── hammersley.py         
├── experiments/           
│   ├── base_cst_model/ 
│   └── base_cir_model/   
├── requirements.txt      
└── notebooks/           
    ├── analyze_constantC_results_3D.ipynb
    └── analyze_CIR_results_4D.ipynb
```

- **`eco_hjb_cstC.py`**: solver for the 3D constant-\(C\) model.  
- **`eco_hjb_cirC.py`**: solver for the 4D CIR-\(C\) model.  
- **`eco_hjb_cirC_reduced.py`**: simplified CIR solver with fewer hyperparameters.  
- **`DGM.py`**: implementation of the DGM architecture used for value and control networks.  
- **`hammersley.py`**: Hammersley sequence generator for residual-based sampling.  
- **`experiments/`**: example trained models with high convergence.  
- **`notebooks/`**: tools for analyzing trained models (value and control surfaces, residuals, trajectories).  

---

## Installation

```bash
git clone https://github.com/MWajdi/eco-hjb-solver
cd eco-hjb-solver

# Create python virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Python ≥ 3.8 and TensorFlow ≥ 2.9 are required.

---
# How to use

## `eco_hjb_cstC.py`: Constant C model solver ((t,P,Y) state)

### Quickstart

```bash
python eco_hjb_cstC.py \
  --mu 0.05 --sigma 0.10 --C 2.0 --a 0.05 --b 0.20 \
  --r 0.05 --T 1.0 --L 0.5 \
  --n_trials 50 --steps_per_trial 1000 --final_training_steps 5000 \
  --candidate_size 100000 --seed 3 --dtype float64 \
  --experiment_name base_cst_model
```

### Command-line arguments

- **Model parameters**
  - `--mu`: drift of the production process.  
  - `--sigma`: volatility of the production process.  
  - `--C`: constant emissions multiplier.  
  - `--a`: base emissions per unit time.  
  - `--b`: marginal emissions per unit of production.  
  - `--r`: discount rate.  
  - `--T`: time horizon.  
  - `--L`: audit threshold for cumulative emissions.  

- **Training parameters**
  - `--n_trials`: number of Optuna trials for hyperparameter tuning.  
  - `--steps_per_trial`: training steps per Optuna trial.  
  - `--final_training_steps`: training steps with the best hyperparameters.  
  - `--candidate_size`: size of the Hammersley candidate pool for residual sampling.  
  - `--seed`: random seed (applied to Python, NumPy, TensorFlow).  
  - `--dtype`: numeric precision (`float32` or `float64`).  
  - `--experiment_name`: tag for experiment outputs.  

The computational domain is automatically set as:
$`
t \in [0,T], \quad
P \in [0.1,10], \quad
Y \in [0,\,C\,(a+b\cdot 10)\,T].
`$

---

### Model and Equation

The file trains neural networks to approximate the solution of the Hamilton–Jacobi–Bellman (HJB) equation in the constant-\(C\) model:

- **Production dynamics**  
$`
\frac{dP_t}{P_t} = \alpha_t \left(\mu\,dt + \sigma\,dW_t\right),
`$

- **Cumulative emissions**  
$`
dY_t = C\,(a+bP_t)\,dt,
`$

- **Running payoff**  
$`
\pi(P) = \sqrt{P},
`$

- **Terminal penalty**  
$`
V(T,P,Y) = -\max(0,\,Y-L).
`$

The associated HJB equation for the value function $`V(t,P,Y)`$ is
$`
\frac{\partial V}{\partial t} + \sup_{\alpha}\left\{\mu \alpha P V_P + \tfrac{1}{2}\sigma^2 \alpha^2 P^2 V_{PP}\right\} + C(a+bP) V_Y + \pi(P) - r V = 0.
`$


---

### Numerical Method

The implementation combines several advances in neural PDE solvers:

- **Deep Galerkin Method (DGM)**  
  Sirignano & Spiliopoulos (2019), *DGM: A deep learning algorithm for solving partial differential equations* (arXiv:1912.01455).  

- **Extensions of the Deep Galerkin Method (DGM-PIA)**  
  Al-Aradi, Correia, de Freitas Naiff, Jardim & Saporito, *Extensions of the Deep Galerkin Method*.  
  Introduces the **Policy-Iteration-Auxiliary (PIA)** scheme, alternating between value and control updates.  

- **Residual-based adaptive sampling**  
  Nabian, Gladstone & Meidani (2022), *Efficient training of physics-informed neural networks via importance sampling* (arXiv:2207.10289).  
  Candidate points are generated via a Hammersley sequence and sampled according to residual magnitudes.  

---

### Training Algorithm

#### Neural networks and targets

- **Value network**: $`f_\theta(t,P,Y)`$ approximates the scaled value $`V(t,P,Y) = s_V \, f_\theta(t,P,Y)`$, with $`s_V > 0`$ (parameter `V_scale`).
- **Control network**: $`g_\phi(t,P,Y)`$ approximates the scaled control $`\alpha(t,P,Y) = s_\alpha \, g_\phi(t,P,Y)`$, with $`s_\alpha > 0`$ (parameter `alpha_scale`).

Gradients are taken w.r.t. the **unscaled** network output and mapped to original units via the scales $`s_V`$ and $`s_\alpha`$.

The residual is computed in **scaled variables** using $`f_\theta`$ and $`g_\phi`$. Denoting
$`
V = s_V f_\theta,\quad
\alpha = s_\alpha g_\phi,
`$
the pointwise **PDE residual** is
$`
\mathcal{R}_{\mathrm{PDE}}(t,P,Y;\theta,\phi)
= \partial_t V + \mu \alpha P\, \partial_P V + \tfrac{1}{2}\sigma^2 \alpha^2 P^2 \, \partial_{PP} V + C(a+bP)\, \partial_Y V + \pi(P) - r V. `$

#### Loss functions

- **PDE loss** (mean-squared residual over interior sample $`\mathcal{S}`$):
$`
\mathcal{L}_{\mathrm{PDE}}(\theta,\phi)
= \frac{1}{|\mathcal{S}|}\sum_{(t,P,Y)\in \mathcal{S}}
\big(\mathcal{R}_{\mathrm{PDE}}(t,P,Y;\theta,\phi)\big)^2.
`$

- **Terminal loss** (samples $`\mathcal{S}_T`$ uniformly at $`t=T`$):
$`
\mathcal{L}_{\mathrm{term}}(\theta)
= \frac{1}{|\mathcal{S}_T|}\sum_{(P,Y)\in \mathcal{S}_T}
\Big( V(T,P,Y) + \max(0,\,Y-L) \Big)^2.
`$

- **Concavity (second-derivative) penalty** (enforces $`V_{PP}\le 0`$):
$`
\mathcal{L}_{\mathrm{conc}}(\theta)
= \lambda_{\mathrm{conc}}
\cdot
\frac{1}{|\mathcal{S}|}
\sum_{(t,P,Y)\in \mathcal{S}}
\Big(\max\{0,\; \partial_{PP} V(t,P,Y)\}\Big)^2,
`$
with a large coefficient $`\lambda_{\mathrm{conc}}`$.

- **Value objective**:
$` \mathcal{L}_{V}(\theta;\phi) = \mathcal{L}_{\mathrm{PDE}}(\theta,\phi) + \mathcal{L}_{\mathrm{term}}(\theta) + \mathcal{L}_{\mathrm{conc}}(\theta). `$

- **Control target** (analytic maximizer of the Hamiltonian under $`V_{PP}<0`$):
$`
\alpha^*(t,P,Y)
= -\frac{\mu\, V_P(t,P,Y)}{\sigma^2\, P\, V_{PP}(t,P,Y)}.
`$
To avoid division by near-zero, the implementation uses a **clamped** second derivative
$`
\widetilde{V}_{PP} = \mathrm{clamp}(V_{PP};\; |V_{PP}|\ge \varepsilon)\quad (\varepsilon>0)
`$
with sign preserved, and defines
$`
\alpha^*_{\mathrm{safe}} = -\frac{\mu\, V_P}{\sigma^2\, P\, \widetilde{V}_{PP}}.
`$

- **Control loss** (Huber regression to the target):
$`
\mathcal{L}_{\alpha}(\phi;\theta)
= \frac{1}{|\mathcal{S}|}
\sum_{(t,P,Y)\in \mathcal{S}}
\mathrm{Huber}_\delta\Big(\alpha(t,P,Y;\phi) - \alpha^*_{\mathrm{safe}}(t,P,Y;\theta)\Big).
`$

#### Residual-based importance sampling

Let $`\mathcal{C}`$ be a Hammersley candidate set in the domain. At a resampling iteration, define the **composite residual score** at a candidate point $`x=(t,P,Y)`$:
$`
S(x) \;=\;
\big|\mathcal{R}_{\mathrm{PDE}}(x)\big|
\;+\; \big| \mathrm{TermErr}(x_T)\big|
\;+\; \big| \mathrm{ConcErr}(x)\big|,
`$
where $`\mathrm{TermErr}(x_T)`$ is the terminal error evaluated at the paired terminal point and $`\mathrm{ConcErr}(x)=\max\{0, V_{PP}(x)\}^2`$. With a schedule exponent $`k`$ and a small $`\varepsilon>0`$, the **sampling probability** for $`x_i\in\mathcal{C}`$ is
$`
\mathbb{P}(x_i)
= \frac{\big(S(x_i)+\varepsilon\big)^{k}}{\sum_{j}\big(S(x_j)+\varepsilon\big)^{k}}.
`$

#### Procedure

For each training step $`s=1,\dots,S`$:

- **If** $`s \bmod \texttt{resample\_every} = 1`$:
  1. Evaluate $`\mathcal{R}_{\mathrm{PDE}}`$, terminal errors, and $`V_{PP}`$ on all candidates $`\mathcal{C}`$.
  2. Form probabilities $`\mathbb{P}(x_i)`$ as above and draw an interior batch $`\mathcal{S}`$.
- Draw a terminal batch $`\mathcal{S}_T`$ uniformly at $`t=T`$.
- **Value update**: take one optimizer step on $`\mathcal{L}_{V}(\theta;\phi)`$.
- **Control update**: recompute $`\alpha^*_{\mathrm{safe}}`$ from current $`\theta`$, then take one optimizer step on $`\mathcal{L}_{\alpha}(\phi;\theta)`$.
- Apply gradient clipping; if NaN/Inf is detected, rollback to the best snapshot and continue.

---

### Hyperparameter Optimization

Hyperparameters are tuned with Optuna in a **multi-objective problem**:
$`
\min \big(\;
\mathcal{L}_{\mathrm{PDE}},
\; \mathcal{L}_{\mathrm{term}},
\; \mathcal{L}_{\mathrm{conc}},
\; \mathcal{L}_{\alpha}
\;\big).
`$

#### Search space (illustrative)

- **Learning rates and schedules**
  - Base rate $`\eta_{\mathrm{base}}`$ (log-uniform) and a split parameter $`\Delta`$:
    $`\eta_f = \eta_{\mathrm{base}}\cdot 10^{+\Delta}`$, $`\eta_g = \eta_{\mathrm{base}}\cdot 10^{-\Delta}`$.
  - Exponential decays: steps and rates for value/control, $`(\tau_f,\gamma_f)`$, $`(\tau_g,\gamma_g)`$.
- **Sampling / batching**
  - Batch size $`B`$, resampling frequency `resample_every`.
  - Residual focusing schedule: $`k \in [k_{\min},k_{\max}]`$ with a linear ramp over $`K_{\mathrm{sched}}`$ steps.
- **Network size**
  - Depth $`L`$ and width $`W`$ of DGM blocks.
- **Output scales**
  - $`s_V`$ and $`s_\alpha`$ (value/control scales).

Each trial optimizes until $`\texttt{steps\_per\_trial}`$ and reports the four objective values. A **lexicographic** selection among non-dominated trials is used to choose a configuration for a longer run with $`\texttt{final\_training\_steps}`$.

**Artifacts.** For each finalized run the code saves:
- snapshot weights at resampling checkpoints (best-so-far),
- full Keras models $`f_\theta`$ and $`g_\phi`$,
- training history (CSV) and metadata (JSON),
- the Optuna study (pickle) to enable analysis and retraining.

## `eco_hjb_cirC.py`: CIR C model solver ((t,P,Y,C) state)

### Quickstart

Run the 4-D CIR model solver with Optuna tuning:

```bash
python eco_hjb_cirC.py --mu 0.05 --sigma 0.05 --a 0.05 --b 0.2 \
  --r 0.05 --T 1.0 --L 0.5 \
  --kappa 2.0 --beta 1.5 --delta 0.3 --rho -0.5 \
  --C_min 0.1 --C_max 5.0 \
  --n_trials 50 --steps_per_trial 1000 --final_training_steps 5000
```

### Arguments

Key CLI arguments:

- `--mu, --sigma` – drift and volatility of $`P_t`$.  
- `--a, --b` – linear coefficients in the $`Y`$ dynamics.  
- `--r` – discount rate.  
- `--T` – time horizon.  
- `--L` – terminal liability.  
- `--kappa, --beta, --delta` – CIR parameters for $`C_t`$: mean reversion speed, long-term mean, and volatility.  
- `--rho` – correlation between $`P_t`$ and $`C_t`$.  
- `--C_min, --C_max` – domain bounds for $`C_t`$.  
- `--n_trials, --steps_per_trial, --final_training_steps` – Optuna hyperparameter search configuration.  
- `--dtype` – `float32` or `float64`.

### Model Equation

Here the control problem is 4-dimensional, with $`C_t`$ following a CIR process:

$`
dC_t = \kappa(\beta - C_t)\,dt + \delta \sqrt{C_t}\,dB_t.
`$

The HJB PDE becomes

$` \begin{aligned} 0 =& \; \partial_t V + \mu \alpha P V_P + \tfrac{1}{2}\sigma^2 \alpha^2 P^2 V_{PP} \\ &+ \rho \sigma \delta \alpha P \sqrt{C}\, V_{PC} + C(a+bP) V_Y \\ &+ \kappa(\beta - C) V_C + \tfrac{1}{2}\delta^2 C V_{CC} + \pi(P) - rV,
\end{aligned}
`$

with terminal condition $`V(T,P,Y,C) = -\max(0,\,Y-L)`$.

### Algorithm

The training algorithm, loss definitions, and residual-based sampling are **the same as in `eco_hjb_cstC.py`**, with the only change being the extended residual that includes the CIR drift/diffusion in $`C_t`$ and the mixed derivative $`V_{PC}`$.  

