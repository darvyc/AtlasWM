# AtlasWM Architecture

## Encoder

Each frame is encoded independently. Trajectory dimensions are flattened only to share computation; every normalization operation acts within one sample. The latent projection head is a per-sample MLP.

## Predictor

The predictor is a causal transformer. At token `t`, action `a_t` modulates the latent token used to predict `z_(t+1)`. Training and planning use the same alignment.

## Rollout contract

For `T0` context states, exactly `T0-1` historical actions connect them. The next proposed action completes the action sequence aligned with the context latent sequence. A multi-frame rollout without these actions is undefined and rejected.

## Regularizer

AtlasReg consumes the flattened set of latent vectors from a minibatch. The regularizer constrains the marginal latent distribution; it does not itself establish that latents contain the task state.

## Planner

CEM samples bounded action sequences, rolls them through the predictor, scores terminal latent distance to an encoded goal, selects elites and exponentially smooths the fitted mean and standard deviation. The best sampled sequence is retained independently of the final fitted mean.

## Data

Trajectory windows preserve action-state alignment. Training, validation and test splits are produced from complete trajectories. NPY stores are memory mapped; manifest datasets concatenate shards lazily.
