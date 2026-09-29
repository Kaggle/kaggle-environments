# Vendored from gymnax 0.0.8 (https://github.com/RobertTLange/gymnax, Apache-2.0).
# Importing gymnax.environments runs its package __init__, which imports bsuite
# and thus matplotlib; that breaks when matplotlib is built for NumPy 1.x.
