#set page(paper: "a4")

#let updatedDate = datetime(
  year: 2025,
  month: 2,
  day: 17
)

#set document(
title: [HD Glue-TTR],
description: [Specification of a vector-symbolic architecture for grounded language understanding],
author: "bill noble", 
date: updatedDate,
) 


#title()
Updated: #updatedDate.display("[Month repr:short] [Day], [Year]")

The goal of this document is to formalize a Vector Symbolic Architecture (VSA) that
connects neural network perceptual representations to a Type Theory with Records (TTR).
The overall strategy is to re-encode 
real-valued neural network-produced vectors as Binary Spatter Code (BSC) vectors using
a modified version of the "HD-Glue" method of #cite(<sutor_gluing_2022>, form:"prose").
We then define a "is instance"  relation ($:$) 
between record types and perceptual inputs
based on a learned model, which is itself simply a 
hyperdimensional BSC vector.

= Modified TTR syntax

The TTR is modified (e.g., from #cite(<cooper_perception_2023>, form:"prose"))
to cater to the VSA coming later. 
Manily we just leave out things that aren't used yet (function types, etc.).
The biggest deviation from standard TTR is the treatment of PTypes---
predicates ar basic types and the PType is a special record type...
It's possible this simplification isn't tennable in the long term but for now it 
means there are fewer cases to consider in the vector semantics.

In this mini-TTR, a type system signature looks like this:

$ chevron.l B, op("Arity"), Lambda, rho, Pi chevron.r $,

where:

- $"B" = {a_1, ..., a_n}$ is a set of basic types
- $Lambda = {l_1, ..., l_lambda}$ is a finite (!) set of labels
- $rho$ is a special "predicate" label
- $Pi = {pi_1,...,pi_m}$ is special seqence of argument labels 
- $op("Arity"): B -> [0, m]$ is a function that tells us the arity of predicate types


#set math.mat(delim: "[")
$ T_"cat" = mat(
  x, :, op("Ind"); 
  c, :, mat(
    rho, :, op("Cat");
    pi_1, :, x
  );
) --> mat(
  x, :, op("Ind"); 
  c, :, op("Cat")(x);
) $



= VSA

Due to the use of HD Glue, we are targeting a different hyperdimensional vector space
from #cite(<cooper_ttr_2023>, form:"prose") (BSC vs HRR),
but we try to re-use notation when possible 
(e.g., using $sigma$ for vectorize functinos).

= Related work

- @korchemnyi_symbolic_2024 - uses paired CLEVR images to learn disentangled represnetations -- need to read again to see how closely this relates to what we're doing
- @laube_qavsa_2024 

#bibliography(title:"References", style: "apa", "vsa-ttr.bib")
