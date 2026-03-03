= The question type subsumption in SPA/TTR 

=== Jonathan (1/28)

In the tie-in from TTR to SPA we have not discussed how to preserve TTR's type subsumption (partial/pre/...) ordering. In other words, we don't have an obvious notion of semantic strength/entailment etc.

What is our story concerning this? I wonder if one cd think in the following terms: we map judgments in TTR to neural conditions in SPA so...

$ tau_1 subset.eq.sq tau_2 $

*implies*

$ sigma(tau_1) "leads to/requires" sigma(tau_2) $

So e.g.,

$ mat(delim: "[", 
  x, :, "ind"; 
  c_1, :, "lion"(x)
) $


This means that neural activity corresponding to $tau_1$ involves neural activity corresponding to $tau_2$.

=== Robin (1/29)

This was something that held true in the proposal in my earlier Jolli paper 
(for subtyping of record types involving additional fields, at least).  
I think it was discussed explicitly there.  
In VSA it has to do with being able to recover the labels from a vector encoding a record type.  
If you can do that then presumably v1 encodes a subtype of v2 
  if for each encoded label in $v$ in $v_2$ the $v_1 * v^(-1)$ 
  encodes a subtype of the type encoded by $v_2*v^(-1)$ where these are record type encodings 
  and $v_1*v^(-1)≈v_2*v^(-1)$ if they are not.  
It’s not at all clear to me if you could get a bunch of spiking neurons to check this relation 
on the VSA approach...

I included Bill in the cc since this is an issue related to our project.

Best,
Robin

=== Andy (1/29)

As far as I can see, the HBB book says nothing about entailment, but it has a section on inference as a challenge (p. 374 ff.). Two quick thoughts: 

#set enum(numbering: "(i)")
  + one could try to "axiomatize" subsumption rules in terms of "if-then" clauses, following the basal ganglia function in Fig 7.18, p. 281. 
  + Perhaps the desired result is already achieved, due to "underspecified" convoluted representation? Suppose all individuals are modelled as $["btype" * "IND" + c * "lion"]$, $["btype" * "IND" + c * "quail"]$, $["btype" * "IND" + c * "Bob"]$ etc pp. 
Then $["btype" * "IND"]$ is just a 
compressed semantic respresentation (aka pointer) which potentially "abbreviates" all the previous structures; and the individual INDs all involve btype IND, of course. 

best,
Andy

=== Jonathan (1/29)



Hi Andy,


I like your idea that the desired result is already achieved, due to "underspecified" convoluted representation?

which strikes me on the right track and going in the same direction as Robin's comments re labels of RTs.


It seems to me that this is essentially though pointing in a topological direction that needs to be made explicit, along the lines of intervals or neighbourhoods of vectors (a bit like the construction of real numbers by sequences of converging rationals---a more extended sequence entails a less extended sequence or some such).


Whet her the folk using quantum logic (Coecke et al) have already done this, who knows...


J

=== Staffan (1/30)

Hello,

Below is what we wrote about subtyping in our Naloma 2023 paper, a bit hard to read but see section 4.10 of the paper. There is also a comment by Andy. 

#block([
I’m not convinced that "the above solution does not work because (27) holds only if $bold(T_1) = bold(T_2)$". After all, it is an axiom (or maybe a theorem) of VSAs that $a+b tilde.op a$. Of course the similarity of $T_1+T_1$ to $T_1$ will be bigger than the similarity of $T_1+T_2$ to $T_1$, since $T_1+T_1=T_1$ (equals, not $tilde$) but that does not invalidate the axiom. However, as Andy commented, further work is needed. 
])

Just so we don’t forget that we already thought about this before 😊

Best,
Staffan

#block(inset:8pt, fill:luma(230), [
Since subtyping can be defined in terms of a TTR equality between two types, this could appear to be a means of formulating the corresponding SPA-TTR definition:

$ 
T_1 subset.sq.eq T_2 "iff" T_1 and.dot T_2 = T_1 tilde.op (bold(T_1 + T_2)) approx T_1 \
sigma(T_1 subset.sq.eq T_2) = (bold(T_1) + bold(T_2) dot.op bold(T_1))
$
])

however th

#line(length: 100%)

Just TTR first... when is $T'$ a subtype of $T$?

Basic types ... here we need to appeal to some model / semanics. Let's ignore for now but at least we have $T subset.eq.sq T$ for all $T$.

Now for record types:

$ T' subset.eq.sq T "iff" forall l in L(T): l in L(T') "and" T'.l subset.eq.sq T.l $

