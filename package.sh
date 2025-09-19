rm -rf ./dist
rm -rf ./build
rm -rf ./kaggle_environments.egg-info
# Delete pycache, pyc, and pyo files
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
# You will first need to run `flit init` once.
flit build
