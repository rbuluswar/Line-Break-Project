# Next steps

1. **Examine how the previous two layers construct these representations.**
   - Track the activation geometry before and after attention/MLP sublayers.
   - Identify when phase, line number, and boundary information emerge.

2. **Explain why phase-only and phase-plus-one-line-vector approximations preserve newline recall but not newline precision.**
   - Compare the activation and logit changes for true newline versus non-newline targets.
   - Inspect whether the approximations cause overprediction of newline tokens or erase competitor-token information.

3. **Examine the role of the other attention heads in the final layer.**
   - Compare their attention patterns, Q/K geometry, and effects on newline precision and recall.
   - Test single-head ablations and interactions among heads.

4. **Work on writing up the results.**
   - Organize the representation, trajectory, intervention, and attention-head findings.
   - Identify the central claims, supporting analyses, limitations, and figures needed for a coherent narrative.
