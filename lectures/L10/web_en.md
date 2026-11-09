<!--
Web version of Lecture 10 (EN). Built into web/L10.html by lectures/common/build_web_lecture.py (format: see its docstring).
Frame hashes: python3 build_web_lecture.py 10 --frames
-->

## @intro

Some business decisions change the situation in which the next decision is taken. A battery that charges now cannot charge again in an hour; a trader who sells today pays a cost to buy back tomorrow; a desk that hedges an option adjusts the hedge every day. Reinforcement learning (RL) is the branch of machine learning built for such sequential decisions: an agent observes a state, takes an action, receives a reward and moves to a new state, and it learns a policy that maximises the total reward.

This lecture writes three real business problems in that language and solves them on real data. The first is the dispatch of a 2 MWh battery on Romanian day-ahead prices, solved exactly by dynamic programming (DP) and then learned by tabular Q-learning and a deep Q-network (DQN). The second is trading the S&P 500 with proximal policy optimisation (PPO), an actor-critic method. The third is hedging a one-month option with a neural policy that minimises the expected shortfall (ES) of the loss. Every agent is trained with five seeds and compared with the rule a practitioner already uses. The verdict is mixed, and instructive: the simple rule wins for the battery, PPO's advantage in trading is not statistically established, and the learned hedge beats the textbook delta in every seed.

@goals
- **Write** a business decision as a Markov decision process (MDP): state, action, reward, transition and discount.
- **Solve** a small MDP exactly with the Bellman equation and dynamic programming, and read the value function.
- **Train** value-based agents (Q-learning, DQN) and a policy-gradient agent (PPO) on real prices, with several seeds and an honest benchmark.
- **Hedge** an option with a neural policy that minimises the expected shortfall of the loss, and compare it with the Black-Scholes delta.
- **Judge** when RL adds value in business and when a simple rule is enough.

# Sequential decisions in business

Reinforcement learning is about what to do, not about what will happen. This section sets up the agent-environment loop and the three problems of the lecture.

## Reinforcement learning and supervised learning {h=d1cebedc}

Supervised learning has a label for every example, and its predictions do not change the next example. In reinforcement learning the agent sees a state $s_t$, chooses an action $a_t$ and receives a reward $r_{t+1}$; the action changes the next state, and no teacher says what the right action was, only how much reward followed. Storage, trading, hedging, inventory, pricing and credit limits all have this structure. When today's action does not change tomorrow's options, a forecast and a rule are enough.

@practice use RL only when today's action changes tomorrow's options.

## The agent-environment loop {h=24991c89}

The agent follows a policy $\pi_\theta(a\mid s)$; the environment answers each action with a reward and a new state. An episode is one delivery day for the battery, a window of 126 trading days for the trader and the 21-day life of an option for the hedger. All three agents are model-free (they learn from sampled transitions $(s_t,a_t,r_{t+1},s_{t+1})$, never from the law of the market) and offline (they learn from historical data).

@practice write down state, action and reward before training anything.

## Three business applications of reinforcement learning {h=ba650448}

| Problem | State $s_t$ | Action $a_t$ | Reward $r_{t+1}$ |
|---|---|---|---|
| Battery | hour, stored energy, yesterday's prices | charge / idle / discharge | cash flow of the hour (EUR) |
| Trading | past returns, volatility, VIX, trend, current position | flat / long | position × next return − cost |
| Hedging | index level, time to maturity, VIX, previous hedge | hedge ratio in [0, 1] | final profit and loss, judged by its ES |

The data are real: hourly day-ahead prices for Romania (RO), Hungary (HU), Germany–Luxembourg (DE-LU) and Bulgaria (BG) from energy-charts.info, the portal of the Fraunhofer Institute for Solar Energy Systems (ISE), based on data of the European Network of Transmission System Operators for Electricity (ENTSO-E); and daily closes of the S&P 500 and of the Cboe Volatility Index (VIX) from Yahoo Finance, up to 31 October 2026. Each problem is split in time order into training, validation and test periods. The benchmarks are rules a manager already uses: yesterday's optimal plan, buy-and-hold, the 200-day moving average and the Black-Scholes delta.

@practice an RL agent is worth deploying only if it beats the rule the desk already uses, after costs and over several seeds.

## Batteries earn the daily price spread, and the spread has grown {h=717180c4}

A battery buys in the cheapest hours and sells in the dearest, so its income depends on the daily spread between the maximum and minimum price, not on the price level. In Romania the mean daily spread was 47.9 EUR/MWh in 2019, 164.3 in 2025 and 182.8 in 2026 so far; the peak month was July 2024, at 394.4 EUR/MWh. Solar generation now pushes midday prices down, and evening peaks remain.

@chart spread_monthly
Monthly mean of the daily max–min spread of day-ahead prices in Romania and Germany–Luxembourg.

@practice the business case of a battery is a bet on future spreads, not on the price level.

# Markov decision processes and the Bellman equation

An MDP formalises a sequential decision; the Bellman equation breaks it into one-step problems that dynamic programming solves exactly when the model is known.

## The five elements of a Markov decision process {h=9ca75897}

An MDP is $\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma)$: states, actions, transition law $P(s'\mid s,a)$, expected reward $R(s,a)$ and discount factor $\gamma\in[0,1]$. The agent maximises the expected return

$$ G_t=\sum_{k=0}^{T-t-1}\gamma^{k}r_{t+k+1},\qquad \pi^{*}=\arg\max_{\pi}\mathbb E_{\pi}[G_0] $$ {#eq:return}

The Markov property requires the state to contain everything relevant: for the battery, the hour and the stored energy; for the trader, the current position, because it determines the cost of the next trade. The battery episodes last 24 steps and use $\gamma=1$; trading uses $\gamma=0.99$.

@practice a missing state variable, such as the current position, makes the agent erratic.

## Value functions measure how good a state is {h=cd29f465}

The state value $V^{\pi}(s)=\mathbb E_\pi[G_t\mid s_t=s]$ is the expected future reward from a state; the action value $Q^{\pi}(s,a)$ is the same after a given first action, and the advantage $A^{\pi}(s,a)=Q^{\pi}(s,a)-V^{\pi}(s)$ compares an action with the policy's average. Value-based methods (Q-learning, DQN) estimate $Q^*$ and act greedily; policy-based methods (REINFORCE, PPO) parameterise the policy directly and use $V$ only as a baseline. For the battery, $V_h(s)$ is the number of euros still to be earned today from hour $h$ with $s$ MWh stored.

## The Bellman equation: recursive decomposition of the value function {h=44664c8b}

The best value now is the best immediate reward plus the discounted best value of the next state:

$$ V^{*}(s)=\max_{a}\Big\{R(s,a)+\gamma\sum_{s'}P(s'\mid s,a)\,V^{*}(s')\Big\},\qquad Q^{*}(s,a)=R(s,a)+\gamma\,\mathbb E_{s'}\big[\max_{a'}Q^{*}(s',a')\big] $$ {#eq:bellman}

For $\gamma<1$ the right-hand side is a contraction, so value iteration converges to a unique fixed point; on a finite horizon one backward pass is exact (Bellman, 1957). Every algorithm of this lecture is an approximate way to solve this equation when the model is unknown or the state too large for a table.

## The battery as an MDP {h=b179343d}

The state is the hour and the stored energy $s_h\in\{0,1,2\}$ MWh; the action $a_h\in\{-1,0,+1\}$ charges, idles or discharges one MWh of stored energy; the reward is the cash flow of the hour:

$$ r_h(a_h)=\begin{cases}\eta_d\,p_h & a_h=+1\ \text{(sell)}\\ -p_h/\eta_c & a_h=-1\ \text{(buy)}\\ 0 & a_h=0\end{cases},\qquad s_{h+1}=s_h-a_h $$ {#eq:battery}

where $p_h$ is the day-ahead price of hour $h$ and $\eta_c=\eta_d=\sqrt{0.85}$, so a full cycle returns 85 % of the energy. The battery starts each day empty, cannot charge when full or discharge when empty (action masking), and is a price taker without degradation cost or grid fees. The data cover 4,292 Romanian delivery days with 24 complete hours, 2 January 2015 to 31 October 2026; since October 2025 each hour is the mean of four 15-minute prices.

@practice the stored energy is what makes the decision sequential.

## Dynamic programming solves the battery day exactly {h=37a314e6}

Going backwards from midnight, with $V_{24}(s)=0$,

$$ V_h(s)=\max_{a:\,0\le s-a\le 2}\big\{r_h(a)+V_{h+1}(s-a)\big\},\qquad h=23,\dots,0 $$ {#eq:dp}

and the revenue of the day is $V_0(0)$. All 4,292 days are solved in milliseconds. Because DP uses the day's 24 prices before they are known, its revenue is an upper bound for any real strategy. The naive benchmark computes yesterday's optimal plan and applies it today. Every learning agent is scored by its capture ratio, its revenue divided by the perfect-foresight revenue.

@practice DP gives the ceiling against which every learning agent is scored.

## Optimal battery schedule for one day {h=1da87f3c}

On 30 June 2026, the day of 2026 with the largest spread, prices ranged from 100.65 EUR/MWh at 13:00 to 954.59 EUR/MWh at 20:00. The optimal plan charges in the two cheapest hours around midday and discharges in two evening hours, earning 1,541 EUR.

@chart dp_day
Hourly day-ahead prices in Romania on 30 June 2026, coloured by the optimal action of the battery.

## The value function prices stored energy hour by hour {h=f6c8ce08}

At 12:00 on the same day $V_{12}(0)$ = 1,541 EUR and $V_{12}(2)$ = 1,760 EUR: half of the difference, 109.38 EUR, is the value of one stored MWh, the shadow price of energy in the battery at noon. After the evening peak, stored energy is worth nothing. A good learned $Q$ must reproduce this shape without seeing the day's prices.

@chart dp_value
Value function $V_h(s)$ of the battery problem on 30 June 2026 for 0, 1 and 2 MWh stored.

## Perfect foresight: what a 2 MWh battery could have earned {h=92387171}

With perfect foresight the battery would have earned 28.4 thousand EUR in Romania in 2019, 150.2 in 2022, 111.2 in 2025 and 98.5 in January–October 2026 (302 days with 24 complete hours). In 2025 the same battery would have earned 109.9 thousand EUR in Hungary, 112.9 in Bulgaria and 84.3 in Germany–Luxembourg. These are ceilings: they follow the spreads, not the price level.

@chart dp_revenue_year
Perfect-foresight revenue of a 2 MWh battery by year and bidding zone (2026: January–October).

@practice investors size storage on these numbers; a real operator captures only part of them.

# Value-based learning: Q-learning and DQN

Q-learning and the deep Q-network learn $Q^*$ from samples, without knowing the day's prices in advance.

## Q-learning: learn Q* from samples, without a model {h=176aea0b}

Q-learning (Watkins & Dayan, 1992) moves $Q(s,a)$ towards the reward plus the best value of the next state:

$$ Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\big[r_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\big] $$ {#eq:qlearning}

The bracket is the temporal-difference (TD) error, a learning signal available after every hour. The target uses the greedy action whatever the agent does next, so Q-learning is off-policy. It converges if every pair is visited infinitely often and the step sizes satisfy the Robbins–Monro conditions. Exploration is $\varepsilon$-greedy, with $\varepsilon$ decayed from 1 to 0.05.

## Definition of the state: information available to the agent {h=e2c1ba7d}

The operator plans before the day's prices are known, so the agent's state is $(h,\ \text{stored energy},\ q_h)$, where $q_h$ is the quartile of hour $h$ among yesterday's 24 prices: 288 states and 3 actions. The agent is trained on 1,746 days of 2019–2023 for 40 passes; the epoch with the best revenue on the 2024 validation year is kept, and the 638 test days, 1 January 2025 to 24 October 2026, are used once.

@practice a state that leaks today's prices would give a perfect but useless backtest.

## What tabular Q-learning learned {h=e40b28c0}

With one MWh stored, the agent mostly charges in the early-afternoon solar trough and discharges in the evening; yesterday's ranking shifts the timing. The table can be printed and audited hour by hour, which is an advantage of tabular methods.

@chart q_policy
Action of the tabular Q-learning agent with 1 MWh stored, by hour and by yesterday's price quartile of that hour (first seed).

## From a table to a network: the deep Q-network {h=ebc9ffe5}

The DQN replaces the table by a network $Q_\theta(s,a)$ and regresses it on the TD target (Mnih et al., 2015). Double DQN lets the online network choose the next action and a slowly updated target network evaluate it, which removes the upward bias of a maximum over noisy estimates (van Hasselt, Guez & Silver, 2016):

$$ y=r+\gamma\,Q_{\theta^{-}}\big(s',\arg\max_{a'}Q_{\theta}(s',a')\big),\qquad \mathcal L(\theta)=\mathbb E\big[\ell_{\text{Huber}}(y-Q_\theta(s,a))\big] $$ {#eq:dqn}

Our network has 82 inputs (yesterday's and last week's 24 standardised prices, the weekday, the hour and the stored energy) and two layers of 128 rectified linear units (ReLU). A replay memory of the last four epochs breaks the correlation of consecutive hours.

## Learning curves: validation revenue by training epoch {h=4235aaa0}

Both agents improve quickly, but on the 2024 validation year they stay below yesterday's optimal plan: the best validation revenue per day, averaged over five seeds, is 200 EUR for the DQN and 211 EUR for Q-learning, against 289 EUR for the naive plan.

@chart battery_learning
Validation revenue per day (2024) after each training epoch; line = mean, band = range over five seeds.

## Test years: how much of the perfect-foresight revenue is captured {h=fd0f5c47}

On the test days the perfect-foresight revenue is 202.1 thousand EUR. Yesterday's optimal plan earns 163.1, tabular Q-learning 115.2 and the double DQN 101.3 (means of five seeds).

@chart battery_test_cum
Cumulative revenue on the test days (2025 to October 2026): perfect foresight, the two learning agents and yesterday's optimal plan.

## Battery results on the test years, five seeds {h=449803cd}

| Strategy | Revenue (thousand EUR) | Capture (%) | Range over seeds (%) |
|---|---|---|---|
| Perfect foresight (upper bound) | 202.1 | 100 | – |
| Double DQN | 101.3 | 50.1 | 43.9 – 58.5 |
| Tabular Q-learning | 115.2 | 57.0 | 50.2 – 61.9 |
| Yesterday's optimal plan | 163.1 | 80.7 | – |

A Diebold–Mariano (DM) test on the daily revenue differences between the DQN and the naive plan, with a Newey–West variance, gives $t=-18.99$: the DQN earns 96.8 EUR less per day, and neither agent beats the naive plan in any seed. The rule wins because it optimises on all 24 of yesterday's prices, while the agents learn one average policy from a compressed state. The lesson is not that RL fails, but that it must be benchmarked against the rule in use; richer states (a price forecast, the naive plan itself as an input) are the natural next step.

@practice report the capture ratio with its range over seeds and a test against the naive plan, never a single best run.

# Policy gradients and PPO: trading the S&P 500

Policy-gradient methods parameterise the policy directly. PPO is the most used of them.

## Policy-gradient methods {h=ca41b33d}

The policy gradient theorem (Sutton et al., 2000) gives

$$ \nabla_{\theta}J(\theta)=\mathbb E_{\pi_{\theta}}\Big[\sum_{t}\nabla_{\theta}\log\pi_{\theta}(a_t\mid s_t)\,\big(G_t-b(s_t)\big)\Big] $$ {#eq:pg}

where $b(s_t)$ is any function of the state, a baseline that leaves the gradient unbiased and lowers its variance. REINFORCE (Williams, 1992) uses the Monte Carlo return $G_t$. The gradient needs no derivative of the transition law, so the market can remain a black box, and stochastic policies explore by themselves.

## Actor-critic and the advantage estimate {h=1c9f70ea}

A critic $V_\phi$ predicts the return and the actor learns from the surprise. Generalised advantage estimation (GAE) (Schulman et al., 2016) averages TD errors $\delta_t=r_{t+1}+\gamma V_\phi(s_{t+1})-V_\phi(s_t)$:

$$ \hat A_t=\sum_{l\ge0}(\gamma\lambda)^{l}\,\delta_{t+l} $$ {#eq:gae}

with $\lambda=0$ giving the one-step TD advantage and $\lambda=1$ the Monte Carlo advantage; we use $\gamma=0.99$ and $\lambda=0.95$. Actor and critic share two layers of 64 tanh units, and an entropy bonus of 0.01 keeps the policy from collapsing too early.

## PPO: clipped policy updates {h=31433f0d}

PPO (Schulman et al., 2017) maximises the clipped objective

$$ \mathcal L^{\text{CLIP}}(\theta)=\mathbb E_t\big[\min\big(\rho_t\hat A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t\big)\big],\qquad \rho_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)} $$ {#eq:ppo}

With $\epsilon=0.2$, once the probability ratio leaves $[1-\epsilon,1+\epsilon]$ in the favourable direction, the gradient of that sample is zero: a cheap substitute for the Kullback–Leibler (KL) trust region of trust region policy optimisation (TRPO). The chart is an analytic illustration of the mechanism, not data.

@chart ppo_clip
The clipped surrogate objective as a function of the probability ratio, for a positive and a negative advantage (analytic function).

## Trading as an MDP: flat or long the S&P 500 {h=373559d0}

Each evening the agent chooses tomorrow's position $a_t\in\{0,1\}$ and receives

$$ r_{t+1}=a_t\ln\frac{P_{t+1}}{P_t}-c\,|a_t-a_{t-1}|,\qquad c=5\ \text{bp} $$ {#eq:trade}

The state contains seven features standardised with 2005–2016 moments only (log returns over 1, 5, 21 and 63 days, the 21-day realised volatility, the VIX and the distance to the 200-day moving average) and the previous position. Each of 150 PPO iterations uses 64 random windows of 126 days from 2005–2016; the iteration with the best Sharpe ratio on 2017–2019 is kept, and the 1,716 test days from 2 January 2020 are used once, for five seeds.

@practice put the cost in the reward, or the agent trades far too often.

## Training and validation Sharpe ratios during PPO training {h=47152809}

The training Sharpe ratio rises with the iterations, while the validation ratio is noisy; buy-and-hold had a Sharpe ratio of 0.95 on the validation years, and the selected iteration is on average the 48th.

@chart trading_learning
Annualised Sharpe ratio of the greedy policy on the last three training years and on the validation years, during training; band = range over five seeds.

## Test period: the PPO agent against two simple rules {h=a26dc429}

On the test period the Sharpe ratio is 0.61 for buy-and-hold, 0.68 for the moving-average rule and 0.83 for PPO, averaged over seeds that range from 0.72 to 0.95.

@chart trading_equity
Value of 1 USD invested on 2 January 2020 in each strategy, after a 5 basis-point cost per change of position; band = range over five PPO seeds.

## Trading results on the test period {h=aaa1ded6}

| Strategy | Return %/yr | Volatility %/yr | Sharpe | Max. drawdown % | Invested % | Trades |
|---|---|---|---|---|---|---|
| Buy and hold | 13.2 | 20.2 | 0.61 | −33.9 | 100 | 1 |
| 200-day moving average | 8.9 | 12.4 | 0.68 | −20.5 | 79 | 41 |
| PPO (mean of 5 seeds) | 15.1 | 17.0 | 0.83 | −23.2 | 82 | 147 |

PPO has the higher Sharpe ratio in all five seeds, but the stationary block bootstrap (Politis & Romano, 1994) of the Sharpe difference between the seed-averaged PPO strategy and buy-and-hold, +0.26, gives a 95 % interval of [−0.08; 0.56]. The interval contains zero: the data do not establish that the agent adds value.

@practice an interval that contains zero means no evidence that the agent adds value, whatever the backtest chart suggests.

## When is the agent out of the market? {h=07ef1e52}

Reading the positions tells what the agent actually does. PPO was invested on 82 % of the test days. Between 20 February and 30 April 2020 it was invested 42 % of the time, against 14 % for the moving-average rule; in 2022 it was invested 63 % of the time, against 15 %. The agent cuts exposure in stress, but less than the trend rule.

@chart trading_exposure
Share of capital invested by the PPO agent (mean over seeds, 21-day mean) and by the moving-average rule.

@practice read the policy, not only its profit: an agent that is simply always long is buy-and-hold with extra model risk.

## Why RL trading backtests mislead {h=3f234657}

Daily equity returns are close to unpredictable, so the agent mostly learns the noise of its training windows. Markets are non-stationary: the training years contain 2008, the test years a pandemic crash, an inflation shock and a long rally. Five seeds already give a range of Sharpe ratios, and reporting the best is backtest overfitting. A price-taking simulator ignores market impact, and costs must be in the reward. RL helps more in problems with a real state and a known reward: order execution, market making, hedging and storage (Hambly, Xu & Yang, 2023).

@practice ask for the seed distribution, the cost assumption and a bootstrap interval before trusting any RL trading result.

# Deep hedging: a policy that minimises tail risk

Deep hedging (Buehler et al., 2019) trains a neural hedging policy on the risk measure the desk reports.

## Hedging a short call: the profit and loss of the desk {h=57b37410}

The desk sells a one-month at-the-money (ATM) call on the S&P 500 and trades the index every day:

$$ \mathrm{PL}(\delta)=C_0+\sum_{k=0}^{20}\delta_k(S_{k+1}-S_k)-(S_{21}-K)^{+}-c\sum_{k=0}^{21}S_k|\delta_k-\delta_{k-1}| $$ {#eq:pl}

where $S_k$ is the index on day $k$ of a real 21-day path (scaled so that $S_0=K=1$), $C_0$ the Black-Scholes premium with the VIX of day 0 as volatility and a zero rate, $\delta_k$ the units held over $(k,k+1]$ and $c=10$ basis points. There is one path per start day: 3,021 for training (2005–2016), 754 for validation and 1,696 for the test (from 2020). On the test paths the mean premium is 2.38 % of the index and the mean payoff 2.53 %, with a mean VIX of 20.7. The paths are real, not simulated, so they keep the jumps and volatility clusters of 2008, 2020 and 2022.

@practice the premium barely covers the mean payoff; the risk desk manages the tail.

## Black-Scholes delta and the deep-hedging objective {h=9f5d1e55}

The benchmark holds the Black-Scholes delta $\delta_k=\Phi(d_1)$, $d_1=[\ln(S_k/K)+\tfrac12\sigma_k^2\tau_k]/(\sigma_k\sqrt{\tau_k})$, with $\sigma_k$ the VIX of day $k$ and $\tau_k=(21-k)/252$. The network $\delta_\theta(\ln S_k,\tau_k,\sigma_k,\delta_{k-1})\in[0,1]$ instead minimises the expected shortfall of the loss, in the form of Rockafellar & Uryasev (2000):

$$ \min_{\theta,w}\ w+\frac{1}{1-\alpha}\mathbb E\big[(-\mathrm{PL}(\delta_\theta)-w)^{+}\big]=\min_\theta\mathrm{ES}_\alpha\big(-\mathrm{PL}(\delta_\theta)\big),\qquad \alpha=0.95 $$ {#eq:es}

At the optimum $w$ is the value at risk (VaR). The profit and loss is differentiable in $\theta$, so the gradient flows back through all 21 days; the previous hedge as an input lets the network learn a no-trade band when costs are high.

@practice the objective is the risk measure the desk reports.

## Hedging profit and loss on the test paths {h=5690221d}

Hedging removes most of the left tail. The standard deviation of the profit and loss is 2.60 % of the index without hedge, 0.74 % with the delta hedge and 0.69 % with the deep hedge.

@chart hedge_pnl
Distribution of the hedging profit and loss on the test paths, in % of the initial index level: no hedge, Black-Scholes delta, deep hedge (first seed).

## Hedging results on the test paths, 10 bp cost {h=049f8b4f}

| Strategy | Mean PL (%) | Std. dev. (%) | ES95 of loss (%) | Turnover |
|---|---|---|---|---|
| No hedge | −0.15 | 2.60 | 6.82 | 0 |
| Black-Scholes delta (VIX) | 0.24 | 0.74 | 1.94 | 1.71 |
| Deep hedge (mean of 5 seeds) | 0.27 | 0.69 | 1.15 | 1.32 |

The ES of the deep hedge ranges from 1.10 to 1.24 across seeds and is below the delta hedge in all five seeds, while the deep hedge also trades less.

@practice judge a hedge by the tail of the loss and by its turnover, not by its mean profit.

## What the network learned: a flatter, cost-aware hedge ratio {h=b6af2f8f}

Ten days before maturity, with the VIX at 20, the learned hedge ratio is flatter than the delta around the strike: 0.46 at the money against 0.51 for Black-Scholes. The network has learned to trade less where trading is expensive and the payoff uncertain.

@chart hedge_delta
Hedge ratio of the deep hedge (mean and range of five seeds) and the Black-Scholes delta as a function of moneyness.

## Transaction costs: where the learned hedge helps {h=d0f6f280}

One network was trained for each cost level from 0 to 50 basis points. Without costs the ES95 is 1.75 % for the delta and 0.90 % for the deep hedge; at 50 basis points, 2.76 % and 2.34 %. The learned hedge also adapts to the real, fat-tailed paths, not only to costs.

@chart hedge_costs
ES95 of the hedging loss on the test paths as a function of the transaction cost: Black-Scholes delta and deep hedge (mean of three seeds per cost level).

@practice the value of a learned hedge grows with frictions: illiquid underlyings, wide spreads, energy and commodity hedges.

## Reinforcement learning in production: model risk {h=0ea6629a}

The agent optimises exactly the reward it is given: omit costs, risk or constraints and it will exploit the gap. A historical replay ignores the agent's own price impact and future regimes. Offline evaluation needs one test period used once, several seeds and a test against the rule in use. In production, action masks, position limits, a kill switch and human sign-off limit the damage; a table or a hedge-ratio curve can be audited, and Lecture 11 explains neural policies; and the live capture ratio and turnover should be compared with the backtest every month.

@practice treat an RL policy like any model of the bank: documented, validated independently, monitored, with a fallback rule.

## @end

On real data the three problems give three verdicts: for the battery, the simple rule wins; for trading, PPO's higher Sharpe ratio is not significant; for hedging, the learned policy wins in every seed. Lecture 11 opens the black box and explains what a neural network has learned.
