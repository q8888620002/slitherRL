# RattLe: a Slither.io reinforcement learning agent
##### Based on the previous work of [slither-rl-agent](https://github.com/zachabarnes/slither-rl-agent)

### Installation Instructions 
- Install [Conda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04) 

- Create Conda env
```
conda create --name slither python=3.5
```

- Activate a conda env
```
source activate slither
```

- Install needed packages(meant for ubuntu VM):
```
sudo apt-get update
sudo apt-get install -y tmux htop cmake golang libjpeg-dev libgtk2.0-0 ffmpeg
```

- Install universe, universe installation dependencies
```
pip install numpy
pip install universe

```
- Install codebase and packages
```
git clone https://github.com/q8888620002/slitherRL.git
cd slitherRL
pip install -r requirements.txt
```

- Install [docker](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-16-04) for ubuntu 16.04 **MAKE SURE TO DO STEP 2 AS WELL**

- Restart VM

### Test installation

Run the test agent script
```
python test.py
```
you should see a tiny rendering of the game or "yay" on the command line.

### Train a model

Run the corresponding shell script. For example, to train our Recurrent Q model, run:
```
train_recurrentq.sh
```
