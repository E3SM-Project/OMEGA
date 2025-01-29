(omega-design-eos)=

# Equation of State

## 1 Overview The equation of state relates density to the prognostic state
variables temperature and salinity.  The prognostic temperature, $\theta$, is
either potential temperature or Conservative Temperature, depending on the
chosen equation of state, and  $S$ is the salinity.  Given  that the governing
equations are  non-Boussinesq, the equation of state class or functor will
provide specific volume from the state variables.  It will provide methods for
computing the specific volume, and its first derivatives.

## 2 Requirements

### 2.1 Requirement: calculate specific volume/density
The specific volume and density will be calculated form from prognostic
temperature, salinity, and pressure (ref function `gsw_specvol(sa,ct,p`)).

### 2.2 Requirement: allow a choice of equations of state
Omega-1 should include the option for two EOS options. The first is a linear
eos, for benchmarking against MPAS-O and to use in simplified cases. The other
option should be a well-established version of TEOS-10 (e.g. Roquet 75-term
polynomial). 

### 2.3 Requirement: computational efficiency
The eos code should not be major computational bottleneck (criteria?)

### 2.4 Requirement: abide by license for the GSW Oceanographic Toolbox
The GSW toolbox can only be used without modification, so a Omega-specific port
of certain functions will be required to support Kokkos/GPU enabled
computation.  The toolbox without modifications can still be used for testing.

### 2.5 Requirement: verification 
The Omega implementation should be compared with the test value published in
Roquet et al 2015, and the official GSW Oceanographic Toolbox (supported by
TEOS-10) to verify correctness.

### 2.6 Desired: freezing temperature calculation
Later versions should include calculating the freezing temperature of seawater.

### 2.7 Desired: output in situ temperature
Later versions should include calculating the in situ  temperature needed for
coupling (at surface including at non-zero pressure).

### 2.8 Desired: output first derivatives
The EOS will need to be able to produce alpha and beta (drho/dT and drho/dS)
for linear expansions in the higher-order pressure gradient.

### 2.9 Desired: check value range
Later versions should include a check on the range of ocean properties to
assess use of TEOS-10 (equivalent to the ``ocean funnel'' check).

### 2.10 Desired: check value range
Later versions should include functions to convert between conservative and
potential temperatures, and absolute and practical salinity. These will be used
to convert the initial conditions or compare to ocean states from other sources
(MPAS-Ocean, reanalysis, other models etc.) These functions may be used offline
in pre- or post-processing but need to be consistent with the EOS
implementation. 

## 3 Algorithmic Formulation

## 4 Design
The GSW-C toolbox will be incorporated into Omega as a submodule.
In order to abide by the license of the toolbox, the toolbox will only be used
in testing to check our implementation. The coefficients for the Roquet et al.
2015 expansion will be based on the published values (Appendix). A Kokkos
implementation of the function to compute specific volume will be ported to
Omega. The GSW-C toolbox sub-module will serve as a baseline reference for our
ports in unit tests. 

For flexibility in optimizing performance, calls to the EOS function should be
able to support both fine-grained single cell and whole array computation.
Given the limitations with using virtual functions for inheritance on the
device, EOS functions will be provided as functors with a consistent interface.
Functions that use the EOS (e.g. pressure gradient) will have a template
argument for the EOS type. A runtime switch statement is then used to call the
user function with the EOS selected in the configuration file.  This approach
allows the EOS to be called within parallel loops without any inner logic
statements and maintains the flexibility to calculate the EOS for a single cell
so EOS calculations can be fused with other computation.  If performance can be
improved by pre-calculating the EOS, the functors can be used to provide
whole-array calculations functions as well.

### 4.1 Data types and parameters
An `enum class` will be used to specify options for the EOS used for an Omega
simulation:
```c++
enum class EOSType{
   Linear,  /// Linear equation of state
   TEOS10Poly75t.  /// Roquet et al. 2015 75 term expansion
}
```
The user will select the EOS type at runtime in the input yaml file under an
eos section:
```yaml
    eos:
       eosType: 'teos10'
```
### 4.2 Functions
Both the linear and TEOS-10 EOS options will have a functor that provides the
same interface:
```c++
class LinearEOS {
  Real operator()(...);
};
class TEOS10Poly75t {
  Real operator()(...); // same arguments as LinearEOS
};
```

Any function that depends on the EOS will be a template:
```c++
template <class EOSType>
void userOfEOS(EOSType EOS, ...) {
  parallelFor(...) {
    Real Rho = EOS(...);
  }
}
```

Functions that depend on the EOS will be called as follows:
```c++
switch(EOSRuntimeChoice) // from Config
{
  case EOSType::Linear:
    userOfEOS(LinearEOS{}, ...);
    break;
  case EOSType::TEOS10Poly75t:
    userOfEOS(TEOS10{}, ...);
    break;
  ...
}
```

## 5 Verification and Testing

### 5.1 Test: Verification of GSW-C submodule A unit test will verify that the
result of the GSW-C toolbox (used as a submodule, without modifications)
matches the expected value of specific volume published in Roquet et al. 2015
within machine precision. The publication only includes one data point (Sa, Ct,
P) to check. 

### 5.2 Test: Verification of our TEOS-10 75-term polynomial implementation A
unit test will verify that the result of the Omega implementation of the Roquet
et al. 2015 75-term polynomial calculation of the specific volume matches the
expected value of specific volume published in Roquet et al. 2015 within
machine precision. The publication only includes one data point (Sa, Ct, P) to
check. 

### 5.3 Test: Verification of TEOS-10 with GSW-C toolbox A unit test will
verify that the result of the Omega and GSW-C toolbox implementations for the
Roquet et al. 2015 expansion are within machine precision over a range of
conservative temperature, absolute salinity, and pressure ranges.

### 5.4 Test: Verification of linear EOS The linear EOS will be verified
against known values from the MPAS-Ocean model for a range of temperature and
salinity values.
