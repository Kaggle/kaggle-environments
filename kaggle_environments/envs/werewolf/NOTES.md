python kaggle_environments/envs/werewolf/scripts/run.py -c kaggle_environments/envs/werewolf/scripts/configs/run/rule_experiment/standard_DisableDoctorSelfSave_DisableDoctorConsecutiveSave.yaml -o experiment/debug_small -d -a -s -r

python kaggle_environments/envs/werewolf/scripts/run.py -c kaggle_environments/envs/werewolf/scripts/configs/run/rule_experiment/standard_DisableDoctorSelfSave_DisableDoctorConsecutiveSave.yaml -o experiment/debug_small -d -r -s


pytest -s --pdb kaggle_environments/envs/werewolf/test_werewolf.py::test_html_render


pytest -s --pdb kaggle_environments/envs/werewolf/test_werewolf.py::test_html_render

python3 -m http.server 8000