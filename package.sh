rm -rf ./dist
rm -rf ./build
rm -rf ./kaggle_environments.egg-info
# Delete pycache, pyc, and pyo files
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
python3 setup.py sdist
python3 setup.py bdist_wheel --universal