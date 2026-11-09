"""Experiment entry points and reusable runners.

Currently this package exposes the E1/E2 command-line runner in
:mod:`experiments.e1_e2`.  The package exists so tests can import
``CachedDecomposer`` and compare it with ``GroundedSAM`` without executing the
CLI.
"""
