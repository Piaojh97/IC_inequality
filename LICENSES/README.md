# Attribution and publication terms

The information-dynamics simulator is derived from the project's existing simulation code, which carries the MIT licence and copyright notice of FIB LAB, Tsinghua University (2023). The original licence is retained in `model-MIT.txt`.

`code/model/legacy_simulator.py` preserves the population simulator's interaction rules. Its unused `turtle` import is removed to avoid an unnecessary GUI dependency, and a Chinese implementation comment is translated into English. `code/model/run.py` supplies a portable interface, aggregate inputs and in-memory tracing; the original individual trace files are not produced.

The source code builds on the authors' earlier *Human–AI Adaptive Dynamics Drive Emergence of Information Cocoons* work. Retain its citation and attribution when distributing the derived simulator.

No additional blanket licence is specified for the manuscript figures, aggregate statistical data or newly prepared wrapper code.
