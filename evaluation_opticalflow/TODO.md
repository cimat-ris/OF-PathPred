* Since most trajectories are close to a constant velocity regime, we could probably give a try to not only make the decoder learn to produce displacements, but maybe deviations to the input displacement (i.e. a form of residual learning). This would need only to add another output representation and would not need very heavy changes, I feel.

* Calibrate the dropout rates. One option could be to implement a Concrete dropout system and include the concrete parameters as parameters to optimize too.

* Include the posture keypoints, when available.

* Thorough evaluation of some of the design choices:
- Stacked RNN?
- BiLSTM?
- Attention?

* Include evaluation on more datasets:
- SDD
- InD

* Latent variables:
- More explicit variables? E.g.a way point at some point in the future?
- Visualization of the resulting distribution (kde)
