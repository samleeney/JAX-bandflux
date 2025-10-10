# Dust Extinction Models in JAX-bandflux

This document outlines the implementation of dust extinction models (CCM89, OD94, F99) within the JAX-bandflux library. The design prioritizes JAX compatibility, efficiency, and alignment with existing codebase patterns.

## 1. Module Structure

-   **Implementation File:** Dust extinction logic resides in [`jax_supernovae/dust.py`](../jax_supernovae/dust.py).
    -   **Reasoning:** This promotes modularity, keeping dust-related code separate from the primary model implementations (e.g., [`jax_supernovae/salt3.py`](../jax_supernovae/salt3.py)). It enhances maintainability and clarity.

## 2. Function Design

-   **Implementation Style:** Pure functions are used, adhering to JAX's functional programming paradigm. This is optimal for JIT compilation (`@jax.jit`) and other JAX transformations (e.g., `jax.vmap`).
    -   Separate JAX-native functions are implemented for each dust law:
        -   `ccm89_extinction(wave, ebv, r_v)`
        -   `od94_extinction(wave, ebv, r_v)`
        -   `f99_extinction(wave, ebv, r_v)`
    -   These functions take wavelength (`wave`) and dust parameters (e.g., `ebv`, `r_v`) as input and output the extinction in magnitudes (A_lambda).

-   **Interface Design:**
    -   Each dust law function returns a JAX array of A_lambda values.
    -   A helper function `get_dust_law(dust_law_name)` selects the appropriate dust function based on a string name.
    -   In the SALT3 model, dust laws are selected based on a numeric index (`dust_type`).

-   **JAX Compatibility:**
    -   All implementations use JAX NumPy (`jnp`).
    -   Functions are designed as pure functions, suitable for `@jax.jit`.
    -   Vectorization over wavelengths is inherent.

    ```python
    # Actual implementation in jax_supernovae/dust.py
    import jax
    import jax.numpy as jnp
    from functools import partial

    @jax.jit
    def ccm89_extinction(wave, ebv, r_v=3.1):
        """Cardelli, Clayton, Mathis (1989) extinction law."""
        # JAX implementation of CCM89 formula
        # Returns: A_lambda (extinction in magnitudes)
        # ...

    @jax.jit
    def od94_extinction(wave, ebv, r_v=3.1):
        """O'Donnell (1994) extinction law."""
        # JAX implementation of OD94 formula
        # Returns: A_lambda
        # ...

    @jax.jit
    def f99_extinction(wave, ebv, r_v=3.1):
        """Fitzpatrick (1999) extinction law."""
        # JAX implementation of F99 formula
        # Returns: A_lambda
        # ...
        
    @jax.jit
    def apply_extinction(flux, extinction):
        """Apply extinction to flux."""
        return flux * jnp.power(10.0, -0.4 * extinction)
    ```

## 3. Integration with SALT3 Model

-   **Flux Calculation Implementation:**
    -   Dust extinction is applied to the rest-frame spectral energy distribution (SED) *before* integration over the bandpass.
    -   This is implemented in flux calculation functions within [`jax_supernovae/salt3.py`](../jax_supernovae/salt3.py), such as `salt3_bandflux` and `optimized_salt3_bandflux`.
    -   The calculation flow:
        1.  Calculate intrinsic rest-frame SED: `rest_flux = x0 * (m0 + x1 * m1) * 10**(-0.4 * cl[None, :] * c) * a`
        2.  Retrieve dust parameters (`dust_type`, `ebv`, `r_v`) from the `params` dictionary.
        3.  If dust parameters are provided:
            a.  Select the appropriate dust law function based on `dust_type` (0=CCM89, 1=OD94, 2=F99).
            b.  Calculate extinction: `extinction = dust_law(restwave, ebv, r_v)`.
            c.  Apply extinction: `rest_flux = dust.apply_extinction(rest_flux, extinction[None, :])`.
        4.  Proceed with bandpass integration using the (potentially extincted) rest-frame flux.

-   **Parameter Handling:**
    -   The following dust parameters can be added to the `params` dictionary:
        -   `dust_type`: Integer index for the dust law (0=CCM89, 1=OD94, 2=F99).
        -   `ebv`: E(B-V) value for dust extinction.
        -   `r_v`: R_V value (ratio of total to selective extinction), defaults to 3.1 if not specified.
    -   Example: `params = {'z': 0.1, 't0': 58650.0, 'x0': 1e-5, 'x1': 0.0, 'c': 0.0, 'dust_type': 0, 'ebv': 0.1, 'r_v': 3.1}`.
    -   The flux calculation functions use these parameters to conditionally apply the selected dust law.

-   **Functional Approach:** JAX-bandflux uses a functional approach with a flat parameter dictionary, which makes it easy to add new parameters like dust extinction.

## 4. Implementation Details

-   **Dust Laws Implemented:** CCM89, OD94, F99.
-   **Implementation Approach:**
    -   The dust laws are reimplemented directly using `jax.numpy`, ensuring full JAX compatibility.
    -   The implementations are based on the formulas from the `extinction` package and validated against `sncosmo`.
    -   The F99 implementation uses a lookup table approach based on pre-computed values from `sncosmo`.
-   **JAX Compatibility:**
    -   Uses `jnp` for all numerical operations.
    -   Functions are pure and JIT-compilable.
    -   Vectorization is used for efficient computation.

## 5. Testing and Validation

-   **Verification against `sncosmo`:**
    -   Extinction curves (A_lambda vs. wavelength) are compared between JAX-bandflux and `sncosmo` implementations.
    -   Tests ensure numerical agreement within acceptable tolerances.
-   **Test Cases:**
    -   **Individual Laws:** Tests with various `ebv` and `r_v` values.
    -   **Integration:** Comparison of JAX-bandflux bandfluxes (with dust applied) against `sncosmo` bandfluxes.
    -   **Parameter Handling:** Verification of correct parameter passing and interpretation.
    -   **JIT Compilation:** Confirmation that all dust-related computations are JIT-compilable.
-   **Example Code:**
    -   The `examples/dust_extinction_example.py` file demonstrates the use of dust extinction with the SALT3 model.

## Flow Diagram

```mermaid
graph TD
    A[Input: phase, bandpasses, params (incl. z, t0, x0, x1, c, dust_type, ebv, r_v)] --> B{salt3_multiband_flux / optimized_salt3_multiband_flux};
    B --> C{Loop over Bandpasses};
    C --> D[optimized_salt3_bandflux for each band];
    D --> E[Calculate restphase, restwave];
    E --> F[salt3_m0(restphase, restwave)];
    E --> G[salt3_m1(restphase, restwave)];
    E --> H[salt3_colorlaw(restwave)];
    I[Combine M0, M1, ColorLaw to get Intrinsic SED: x0 * (M0 + x1*M1) * 10**(-0.4*CL*c)];
    F --> I;
    G --> I;
    H --> I;
    I --> J{Apply Dust Extinction? (based on params['dust_type'], params['ebv'])};
    J -- Yes --> K[Select Dust Law Function based on dust_type (0=CCM89, 1=OD94, 2=F99)];
    K --> L[Calculate extinction(restwave, ebv, r_v)];
    L --> M[Attenuated SED = apply_extinction(Intrinsic SED, extinction)];
    J -- No --> M;
    M --> N[Integrate SED through Bandpass Transmission];
    N --> O[Apply Zero Point (if any)];
    O --> P[Output: Bandflux];
    C -.-> P;