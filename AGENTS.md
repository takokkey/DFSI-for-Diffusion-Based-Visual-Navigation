# AGENTS.md

## Project intent
This repository is evolving from DFSI (execution-time safety plug-in) toward TPGS-R (task-preserving geometry-guided sampling with reliability conditioning).

The long-term research direction is:
- frozen diffusion-based visual navigation backbone
- runtime local geometry reused from DFSI
- safety and motion consistency moved from output-time filtering toward reverse-denoising-time guidance
- real-robot deployability is a hard requirement

## Engineering priorities
1. Do not break the existing DFSI baseline.
2. Prefer modular additions over large rewrites.
3. Reuse runtime geometry, reliability, logging, and evaluation code whenever possible.
4. Preserve real-robot launch and runtime compatibility.
5. Keep guided and unguided outputs both observable for analysis.
6. Make changes easy to compare in experiments.

## Research priorities
For TPGS-R-v1, prioritize:
- observed geometry guidance
- task-preserving anchor
- prefix / motion consistency
- reliability-conditioned guidance gain

Do not prioritize yet:
- full hidden-risk modeling
- retraining the diffusion backbone
- large framework refactors

## Repository rules
- Treat DFSI as the frozen baseline and keep it runnable.
- Prefer adding new modules for TPGS-R-v1 instead of rewriting DFSI modules.
- Do not change training unless explicitly requested.
- Do not silently change robot control semantics.
- Prefer explicit runtime interfaces for geometry and reliability.
- Preserve reproducibility and experiment logging.

## Code organization guidance
- Keep DFSI baseline code minimally modified.
- Add new TPGS-R-v1 logic in clearly separated modules when possible.
- Reuse DFSI runtime geometry, clearance query, reliability signals, logging, and evaluation utilities.
- Keep DFSI execution-time safety logic available as fallback safety, not as the main TPGS-R method.

## Expected deliverables for TPGS-R-v1 work
- docs/TPGSR_V1_REFACTOR_PLAN.md
- docs/TPGSR_V1_IMPLEMENTATION_SUMMARY.md
- runnable config(s) for:
  - backbone baseline
  - DFSI baseline
  - TPGS-R-v1
- logging support for:
  - guided vs unguided trajectory
  - reliability score
  - safety / task / consistency terms
  - fallback safety trigger

## Review guidelines
- Reject changes that remove or invalidate DFSI baseline comparisons.
- Reject unnecessary changes to robot control semantics.
- Flag edits that tightly couple TPGS-R logic back into DFSI fallback logic.
- Preserve real-robot deployability.
- Preserve experiment comparability.
