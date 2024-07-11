#### Surrogate surface simulation

This folder contains an example of how to construct a so-called surrogate surface of solid-solution alloy. The surface emulates an fcc111 surface via array-based "positions" of atoms. In conjunction with the lGNN model this allows for rapid inference of adsorption energy distributions of extended solid-solution and high-entropy alloy surfaces.

An inter-adsorbate blocking scheme is available for *OH and *O as elaborated upon in the original [publication](https://doi.org/10.1002/advs.202003357). To replicate the same workflow scaling and displacement of the energies are necessary which is supported by the SurrogateSurface class.

Additionally, should it suit your application, there is an option for direct energy input which will override the predicted adsorption energies and assign a single value e.g. the DFT calculated energy, to all surface sites
```python
adsorbates = ['OH','O']
sites = ['ontop','fcc']
dft_values = [dft_OH,dft_O]
surface = SurrogateSurface(composition, adsorbates, sites,..., direct_e_input=dft_values)
```

To simulate scenarios with an overlayer or skin of a certain element a separate alloy composition can be given for each layer in the surrogate surface with a list of dictionaries:
```python
composition = [{'Pt':1.0},              # Surface layer -> Pure Platinum
               {'Pt':0.75, 'Ru':0.25}   # Subsurface layer -> Some Ruthenium depletion
               {'Pt':0.5, 'Ru':0.5}     # Third layer -> Bulk alloy composition
              ]
```
