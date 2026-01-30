# Problem 1: Estimating Weekly Fan Votes

To address the core challenge of estimating unobserved weekly fan votes while ensuring consistency with DWTS's official elimination rules and quantifying estimate certainty, we abandon traditional latent popularity random walks in favor of a **rule-consistent, end-to-end trainable framework**. The key limitations of prior approaches—non-differentiable rank operations, decoupled elimination classifiers, and instability from non-identifiable vote scales—are resolved by integrating a Bayesian network structure with Plackett–Luce (PL) inference (dubbed the PL–BN Model). This model directly trains fan vote shares through the show’s official composite risk scores, uses differentiable proxies for discrete ranks, and leverages the PL model’s native support for listwise tail-event likelihoods (e.g., bottom-1/bottom-2 eliminations). The result is a framework that strictly adheres to DWTS’s rules, produces interpretable vote estimates, and naturally quantifies uncertainty via parameter posterior distributions and prediction intervals.

## PL--BN Model: Rule-Consistent Bayesian Network with Plackett--Luce Inference

In DWTS, weekly eliminations are determined by combining judges' technical scores with audience popularity votes. Fan vote totals are not released. The task requires estimating weekly fan votes for each couple during weeks they remain active, and quantifying (i) **consistency** with observed eliminations and (ii) **certainty** of the estimates.

### Variables Definition

#### Indices
- $s \in \{1, \dots, S\}$: season index ($S=34$),
- $t \in \{1, \dots, T_s\}$: week index within season $s$,
- $i \in \mathcal{A}_{s,t}$: couple index within the *active set* $\mathcal{A}_{s,t}$ (all couples remaining in week $t$ of season $s$).

#### Observed quantities (from the CSV)
- Judges' weekly total score:
  $$J_{i,t} = \sum_{j \in \mathcal{J}_{s,t}} \text{score}_{i,t,j}, \tag{1}$$
  where $\mathcal{J}_{s,t}$ denotes the set of available judges in week $t$ of season $s$.

- Judges' percentage (for percent rule seasons):
  $$P^J_{i,t} = \frac{J_{i,t}}{\sum_{k \in \mathcal{A}_{s,t}} J_{k,t}}, \tag{2}$$
  representing the proportion of total judges' scores obtained by couple $i$ relative to all active couples.

- Judges' rank (for rank rule seasons): $R^J_{i,t} \in \{1, \dots, |\mathcal{A}_{s,t}|\}$,
  defined by ordering $J_{i,t}$ in descending order within $\mathcal{A}_{s,t}$ (1 = highest score). Ties are resolved via a fixed convention (e.g., alphabetical order of couple names).

- Observed elimination set $\mathcal{E}_{s,t} \subseteq \mathcal{A}_{s,t}$ parsed from the `results` field:
  Typically $|\mathcal{E}_{s,t}|=1$ (single elimination), with edge cases of $|\mathcal{E}_{s,t}|=0$ (no elimination) or $|\mathcal{E}_{s,t}|>1$ (multiple eliminations).

#### Unobserved target: fan votes
- Fan vote total: $F_{i,t} \ge 0$ (absolute number of votes received by couple $i$ in week $t$; unobserved).
- Weekly total votes: $V_t = \sum_{k \in \mathcal{A}_{s,t}} F_{k,t}$ (total fan votes cast in week $t$; non-identifiable from elimination outcomes, as only relative vote strength matters).
- Fan vote share (identifiable up to scale):
  $$p^F_{i,t} = \frac{F_{i,t}}{V_t}, \quad \sum_{i \in \mathcal{A}_{s,t}} p^F_{i,t} = 1, \tag{3}$$
  the core identifiable quantity for inference (absolute votes are recovered via $\widehat{F}_{i,t} = \widehat{V}_t \cdot p^F_{i,t}$ using a chosen scaling factor $\widehat{V}_t$).

#### Key modeling change (new variables)
Instead of introducing a separate latent "popularity random walk" state as the main generator, we introduce **weekly fan-vote utilities** $a_{i,t} \in \mathbb{R}$ as the fundamental latent variables to be learned. These utilities map to valid vote shares via the softmax function (ensuring non-negativity and unit sum):
$$p^F_{i,t} = \mathrm{softmax}\!\big(a_{\cdot,t}\big)_i = \frac{\exp(a_{i,t})}{\sum_{k \in \mathcal{A}_{s,t}} \exp(a_{k,t})}, \tag{4}$$
where $a_{\cdot,t} = \{a_{1,t}, a_{2,t}, \dots, a_{|\mathcal{A}_{s,t}|,t}\}$ denotes the utility vector of all active couples in week $t$.

The utilities are parameterized by a trainable scorer to incorporate observable covariates:
$$a_{i,t} = h_{\theta}(x_{i,t}), \tag{5}$$
where:
- $x_{i,t}$ is a feature vector constructed from available covariates (e.g., celebrity age/industry, professional partner identity, week index, judges' score trends, regional popularity proxies),
- $\theta$ denotes the learnable parameters of the scorer $h_{\theta}(\cdot)$ (e.g., weights of a neural network or linear model),
- This parameterization makes the framework end-to-end trainable while preserving interpretability: vote shares are derived from utility scores, which are learned from observable features, and elimination rules are applied directly to these vote shares.

### Official composite score $S_{i,t}$
We preserve the show-defined composite/risk score as the sole pathway linking judges' scores and fan votes to elimination risk. The formulation varies by season-specific rules.

#### Percent rule (Seasons 3--27)
The composite score is the sum of judges' percentage and fan vote share, with smaller values indicating higher elimination risk:
$$S^{\text{percent}}_{i,t} = P^J_{i,t} + p^F_{i,t}, \quad \text{(smaller $S$ $\Rightarrow$ higher elimination risk)}, \tag{6}$$
directly reflecting the show’s rule of combining technical performance (judges' percentage) and audience support (fan vote share).

#### Rank rule (Seasons 1--2, 28--34)
The composite score is the sum of judges' rank and fan vote rank, with larger values indicating higher elimination risk:
$$S^{\text{rank}}_{i,t} = R^J_{i,t} + R^F_{i,t}, \quad R^F_{i,t} = \mathrm{rank}\big(p^F_{i,t}\big), \tag{7}$$
where $R^F_{i,t}$ is the rank of $p^F_{i,t}$ in descending order (1 = highest vote share).

Since the $\mathrm{rank}(\cdot)$ function is discrete and non-differentiable (hindering gradient-based training), we use a **training-time differentiable proxy** $\widetilde{R}^F_{i,t}$ that converges to the hard rank as a smoothing parameter $\rho \downarrow 0$:
$$\widetilde{R}^F_{i,t} = 1 + \sum_{k \in \mathcal{A}_{s,t}, \, k \neq i} \sigma\!\left(\frac{p^F_{k,t} - p^F_{i,t}}{\rho}\right), \quad \sigma(x) = \frac{1}{1 + e^{-x}}, \tag{8}$$
where $\sigma(\cdot)$ is the sigmoid function (outputs $[0,1]$). For each other couple $k$, $\sigma\left(\frac{p^F_{k,t} - p^F_{i,t}}{\rho}\right)$ approximates an indicator variable (1 if $p^F_{k,t} > p^F_{i,t}$, 0 otherwise), so $\widetilde{R}^F_{i,t}$ approximates the hard rank.

The training-time composite rank score is:
$$\widetilde{S}^{\text{rank}}_{i,t} = R^J_{i,t} + \widetilde{R}^F_{i,t}. \tag{9}$$
For final reporting and result replay, we revert to the hard rank rule using $R^F_{i,t} = \mathrm{rank}(p^F_{i,t})$.

#### Bottom-two + judges' save (Seasons 28--34)
This two-step rule first identifies the "bottom-two" couples with the highest composite rank scores, then lets judges eliminate one of them:
1. Bottom-two set identification:
   $$\mathcal{B}_{s,t} = \arg\max_{\{i,j\} \subset \mathcal{A}_{s,t}} \ \widetilde{S}^{\text{rank}}_{i,t} + \widetilde{S}^{\text{rank}}_{j,t}, \tag{10}$$
   selecting the pair of couples with the largest sum of training-time composite scores (highest combined elimination risk).

2. Judges' elimination probability:
   Given $\mathcal{B}_{s,t} = \{a,b\}$, the probability that couple $a$ is eliminated is:
   $$\Pr(a \text{ eliminated} \mid \mathcal{B}_{s,t}) = \sigma\!\left(\kappa \cdot (J_{b,t} - J_{a,t})\right), \tag{11}$$
   where $\kappa > 0$ controls the strength of judges' preference for higher technical scores (larger $\kappa$ = stronger preference for eliminating the lower-scoring couple in the bottom-two).

### Plackett--Luce (PL) training: tail likelihood induced by $S_{i,t}$
We avoid separate "elimination classifiers" and instead train $p^F_{i,t}$ directly through the official composite score, leveraging the Plackett–Luce model \cite{Plackett1975} for a differentiable likelihood over elimination tail events (bottom-1/bottom-$m$). This model:
- Is listwise (considers all active couples, not just pairs),
- Supports varying active set sizes $|\mathcal{A}_{s,t}|$,
- Becomes a hard elimination rule in the low-noise limit ($\tau_r \to \infty$),
- Avoids non-differentiable order-statistics by aligning with the composite score $S_{i,t}$.

#### Rule-specific temperature
We introduce a rule-dependent inverse-temperature $\tau_r > 0$ to control the concentration of the likelihood on elimination tail events:
$$\tau_r = \begin{cases}
\tau_{\text{percent}}, & \text{percent rule seasons}, \\
\tau_{\text{rank}}, & \text{rank rule seasons (including bottom-two stage)}.
\end{cases} \tag{12}$$
Larger $\tau_r$ makes the likelihood more concentrated on the couple(s) with the highest elimination risk (closer to deterministic elimination), while smaller $\tau_r$ allows for more noise in predictions.

#### Rule-aligned tail utility
To unify the direction of elimination risk across rules, we define a tail-utility $u_{i,t}$ that **monotonically increases with elimination risk**:
$$u_{i,t} = \begin{cases}
-\,S^{\text{percent}}_{i,t}, & \text{percent rule (smaller $S$ $\Rightarrow$ higher risk)}, \\
+\,\widetilde{S}^{\text{rank}}_{i,t}, & \text{rank rule (larger $S$ $\Rightarrow$ higher risk)}.
\end{cases} \tag{13}$$
This alignment ensures the PL model consistently prioritizes the correct couples for elimination regardless of the season’s rule.

#### Bottom-1 (single elimination) PL likelihood
For weeks with single elimination ($|\mathcal{E}_{s,t}|=1$) and eliminated couple $e_{s,t}$, the likelihood is:
$$\Pr(e_{s,t} \mid \mathcal{A}_{s,t}) = \frac{\exp\!\left(\tau_r \cdot u_{e_{s,t},t}\right)}{\sum_{k \in \mathcal{A}_{s,t}} \exp\!\left(\tau_r \cdot u_{k,t}\right)}. \tag{14}$$
This quantifies the probability of observing $e_{s,t}$ as the eliminated couple, given the model’s current parameters.

#### Bottom-$m$ (multiple eliminations) PL likelihood (without replacement)
For weeks with multiple eliminations ($|\mathcal{E}_{s,t}|=m$) and observed ordered elimination sequence $\{\ell_1, \dots, \ell_m\}$ (e.g., first eliminated $\ell_1$, then $\ell_2$, etc.), the likelihood is computed as a product of sequential conditional probabilities:
$$\Pr(\ell_1, \dots, \ell_m) = \prod_{q=1}^{m} \frac{\exp\!\left(\tau_r \cdot u_{\ell_q,t}\right)}{\sum_{k \in \mathcal{A}_{s,t} \setminus \{\ell_1, \dots, \ell_{q-1}\}} \exp\!\left(\tau_r \cdot u_{k,t}\right)}. \tag{15}$$
If only an unordered elimination set is observed (e.g., $\{\ell_1, \ell_2\}$ with no order), we marginalize over all permutations of the set (computationally feasible for small $m$).

#### Season 28+ with judges' save (two-step likelihood)
For weeks under the bottom-two + judges' save rule, the total likelihood of eliminating couple $a$ is the product of two probabilities: (1) $a$ is selected into the bottom-two set, and (2) $a$ is eliminated by the judges given the bottom-two set:
$$\Pr(\text{elim}=a) = \Pr(\mathcal{B}_{s,t} = \{a,b\}) \cdot \Pr(a \text{ eliminated} \mid \mathcal{B}_{s,t} = \{a,b\}), \tag{16}$$
where $\Pr(\mathcal{B}_{s,t} = \{a,b\})$ is the bottom-2 PL likelihood (computed via Equation 15 with $m=2$) using $\tau_{\text{rank}}$ and $u_{i,t}$, and $\Pr(a \text{ eliminated} \mid \mathcal{B}_{s,t} = \{a,b\})$ is given by Equation 11.

### Training objective
Let $\theta$ denote the learnable parameters of the scorer $h_{\theta}(\cdot)$ (producing $a_{i,t}$ and hence $p^F_{i,t}$). The model is trained by maximizing the **penalized log-likelihood** of observed elimination outcomes:
$$\max_{\theta} \ \sum_{s,t} \log \Pr(\mathcal{E}_{s,t} \mid \mathcal{A}_{s,t}; \theta) - \lambda \sum_{s,t} \sum_{i \in \mathcal{A}_{s,t}} \big(a_{i,t} - a_{i,t-1}\big)^2 - \gamma \|\theta\|_2^2. \tag{17}$$

The three terms in the objective function are:
1. **Log-likelihood term**: Maximizes the agreement between the model’s predicted elimination probabilities and the observed outcomes (core training signal).
2. **Temporal smoothness regularizer**: Penalizes large changes in fan-vote utilities $a_{i,t}$ between consecutive weeks (i.e., $\big(a_{i,t} - a_{i,t-1}\big)^2$). This stabilizes inference for non-identifiable vote scales by enforcing gradual changes in fan preferences (a realistic assumption, as audience sentiment rarely shifts abruptly).
3. **L2 regularization term**: $\gamma \|\theta\|_2^2$ penalizes large parameter values to prevent overfitting to noisy elimination outcomes or covariates.

All terms are fully differentiable in $\theta$ (via the chain $a_{i,t} \to p^F_{i,t} \to S_{i,t} \to u_{i,t} \to$ PL likelihood), enabling gradient-based optimization (e.g., Adam, stochastic gradient descent).

### References
[^1]: Plackett, R. L. (1975). The analysis of permutations. *Journal of the Royal Statistical Society, Series B (Methodological)*, 37(2), 193–202.