# Mitigation of the Cancellation Problem in Local Gyrokinetic Simulations


# ![alt text](/pictures/evaluation/benchmark/comparison/fields/kthrho0.300_beta0.000-0.022_fields_f-version.png)

## Introduction

This repository is focused on my work for my Master Thesis about the topic of the Mitigation of the Cancellation Problem in Local Gyrokinetic Simulations. This Thesis is based on the works of Paul Crendall.

* [GKW-Code](https://bitbucket.org/gkw/gkw/wiki/Home)
* Master-Thesis: [PDF](masterthesis/Lippert%20-%20Mitigation%20of%20the%20Cancellation%20Problem.pdf)
* Literature: [BIB](masterthesis/literature.bib)

## Abstract

The so-called cancellation problem limits electromagnetic gyrokinetic simulations to low
plasma beta values and long perpendicular turbulent length scales. To mitigate this
problem a new numerical scheme was found for global nonlinear gyrokinetic simulations.
This scheme relies on the introduction of an electromagnetic additional field, the plasma
induction, through Faraday’s law. The implementation of Faraday’s law into GKW results
in the calculation of the gyrocenter distribution, which before needed to be modified. For
local linear simulations the computational time increases by 10 %.

With local linear benchmarks the validity of the implementation was proven plausible.
However, the cancellation problem could not be solved by the plasma induction in linear
simulations. Simulations using the nonlinear version of GKW showed the trend that the
implementation is valid, but they could not be finished due to time constraints.
