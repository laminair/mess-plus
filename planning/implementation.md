# Implementation Plan: Fixing The MESS - Practical LLM Routing Service Level Guarantees Under Sparse User Feedback

## MESS+ core mechanism

For each request t:


1. Sample $X_t$ ~ Bernoulli($p_t$) where p_t = min(1, $\frac{c}{\sqrt[4]{t}}$)
    
    1. IF $X_t$ = 1 (EXPLORE):
         - Query ALL models with the request
         - Get actual satisfaction $s_{m,t}$ for each model m
         - Update predictor weights using SGD
         - Return output from best-performing model
    
    2. ELSE (EXPLOIT):
         - Use predictor to get $\hat{s}_{m,t} for all models
         - Solve optimization: $m* = argmin_m [V·E_{m,t} + Q_t·(α - \hat{s}_{m,t})]$
         - Query only model {m*}
         - Get actual satisfaction {s_{m*,t}}
    
    3. Update virtual queue:
         $Q_{t+1} = max\{0, Q_t + \alpha - s_{m*,t}\}$


## Current situation
The first two points are the priority right now. 
Tuning _**V**_ is a potential third paper on the series.

1. The algorithm expects **"readily available user satisfaction labels for any request"** (Section 5)
User feedback may be only sparingly available in production systems, requiring us to adapt the learning mechanism. 

1. During exploration phases (when X_t = 1), the **system queries all models and needs feedback for each**
Showing multiple outputs to users and requesting feedback may be confusing and practically impossible. The current implementation makes exploration expensive and limits the practical applicability of MESS+

1. Can we scale to 10+ models with the next-gen version of MESS+? Sparse feedback should enable us to scale much better at significantly higher speed.

2. Requires **manually setting V** (cost-efficiency tradeoff parameter)
While "coarse tuning works reasonably well," optimal V selection appears task-dependent. No automated mechanism for V selection provided.

1. Low prio - down the road: Can we use MESS+ observations to determine what models to schedule?
