# Cosmological Inference

After training the emulator, the final step in the pipeline is to perform cosmological inference using the observed data. This is done using the [`sunbird`](https://github.com/florpi/sunbird) inference module.

## Inference Methods

The `sunbird.inference` module provides several inference methods:

- **Likelihood-based inference**: Traditional Bayesian inference using MCMC or nested sampling
- **Simulation-based inference (SBI)**: Neural network-based inference methods
- **Approximate Bayesian Computation (ABC)**: Likelihood-free inference

## Running Inference

The general workflow for inference is:

1. Load the trained emulator for your chosen statistic(s)
2. Prepare the observed data in the correct format
3. Define the priors for your cosmological and galaxy-halo parameters
4. Run the inference using the appropriate method
5. Save and analyze the chains

### Example Usage

```python
from sunbird.inference import Inference
from acm.observables import MyObservable  # Your observable class

# Initialize the observable with paths to model and data
obs = MyObservable(...)

# Get the observed data and emulator
data = obs.get_data()
emulator = obs.get_model()

# Set up inference
inference = Inference(
    emulator=emulator,
    data=data,
    priors=priors,
)

# Run inference
chains = inference.run()
inference.save_chain(path='chains/my_chain.pkl')
```

```{seealso}
For detailed examples, see the [sunbird documentation](https://github.com/florpi/sunbird) and the inference scripts in the projects directory.
```

## Chain Storage

Inference chains are stored following the conventions described in the [Data Storage](../code/data#inference-chains) section. 

The chain files contain:
- `samples`: The parameter samples from the chain
- `weights`: Sample weights (if applicable)
- `ranges`: Parameter ranges
- `names`: Parameter names
- `labels`: LaTeX labels for plotting
- `log_likelihood`, `log_prior`, `log_posterior`: (method-dependent)

## Visualization and Analysis

After running inference, you can use `getdist` or other tools to visualize and analyze the chains:

```python
import getdist
from getdist import plots

# Load chain
samples = getdist.loadMCSamples('chains/my_chain')

# Create triangle plot
g = plots.get_subplot_plotter()
g.triangle_plot([samples], filled=True)
```

```{tip}
The `acm.utils` module provides helper functions for chain analysis and visualization.
```
